/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#include <memory>
#include <time.h>
#include <sys/stat.h>
#include <signal.h>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <numeric>

#include <iostream>
#include <sys/time.h>
#include <vector>
#include <stdlib.h>
#include <typeinfo>
#include <ctime>
#include <sys/types.h>
#include <stdint.h>
#include <linux/limits.h>
#include <dirent.h>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "camera/stereo_camera.hpp"
#include "sgm_cuda/disparity_method.h"

static bool is_streaming = true;
static void sig_handler(int sig)
{
    is_streaming = false;
}

static cv::CommandLineParser getConfig(int argc, char **argv)
{
    const char *params = "{ help           | false              | print usage          }"
                         "{ fps            | 30                 | (int) Frame rate }"
                         "{ width          | 1280               | (int) Image width }"
                         "{ height         | 720                | (int) Image height }";

    cv::CommandLineParser config(argc, argv, params);
    if (config.get<bool>("help"))
    {
        config.printMessage();
        exit(0);
    }

    return config;
}

int main(int argc, char *argv[])
{
    // handle signal by user
    struct sigaction act;
    act.sa_handler = sig_handler;
    sigaction(SIGINT, &act, NULL);

    // parse config from cmd
    cv::CommandLineParser config = getConfig(argc, argv);

    StereoCameraConfig camConfig;
    camConfig.fps = config.get<int>("fps");
    camConfig.width = config.get<int>("width");
    camConfig.height = config.get<int>("height");

    StereoCamera::Ptr camera = std::make_shared<StereoCamera>(camConfig);
    if (!camera->checkCameraStarted())
    {
        std::cout << "Camera open fail..." << std::endl;
        return 0;
    }

    // Stereo Camera
    cv::Mat frame_0, frame_1, frame_0_rect, frame_1_rect;
    cv::Mat disp16, disp32;
    StereoCameraData camDataCamera;

    cv::FileStorage fs("ocams_calibration_720p.xml", cv::FileStorage::READ);

    cv::Mat D_L, K_L, D_R, K_R;
    cv::Mat Rect_L, Proj_L, Rect_R, Proj_R, Q;
    cv::Mat baseline;
    cv::Mat Rotation, Translation;

    fs["D_L"] >> D_L;
    fs["K_L"] >> K_L;
    fs["D_R"] >> D_R;
    fs["K_R"] >> K_R;
    fs["baseline"] >> baseline;
    fs["Rotation"] >> Rotation;
    fs["Translation"] >> Translation;

    // Code to calculate Rotation matrix and Projection matrix for each camera
    cv::Vec3d Translation_2((double *)Translation.data);

    cv::stereoRectify(K_L, D_L, K_R, D_R, cv::Size(camConfig.width, camConfig.height), Rotation, Translation_2,
                      Rect_L, Rect_R, Proj_L, Proj_R, Q, cv::CALIB_ZERO_DISPARITY);

    cv::Mat map11, map12, map21, map22;

    cv::initUndistortRectifyMap(K_L, D_L, Rect_L, Proj_L, cv::Size(camConfig.width, camConfig.height), CV_32FC1, map11, map12);
    cv::initUndistortRectifyMap(K_R, D_R, Rect_R, Proj_R, cv::Size(camConfig.width, camConfig.height), CV_32FC1, map21, map22);

    if (MAX_DISPARITY != 128)
    {
        std::cerr << "Due to implementation limitations MAX_DISPARITY must be 128" << std::endl;
        return -1;
    }
    if (PATH_AGGREGATION != 4 && PATH_AGGREGATION != 8)
    {
        std::cerr << "Due to implementation limitations PATH_AGGREGATION must be 4 or 8" << std::endl;
        return -1;
    }

    std::vector<float> times;
    float elapsed_time_ms;

    init_disparity_method(0, 0);

    while (1)
    {
        if (!is_streaming)
        {
            std::cout << "Exit by user signal" << std::endl;
            break;
        }
        if (camera->getCamData(camDataCamera))
        {
            frame_0 = camDataCamera.frame_0;
            frame_1 = camDataCamera.frame_1;

            cv::remap(frame_0, frame_0_rect, map11, map12, cv::INTER_LINEAR);
            cv::remap(frame_1, frame_1_rect, map21, map22, cv::INTER_LINEAR);

            // Compute
            cv::Mat disparity_im = compute_disparity_method(frame_0_rect, frame_1_rect, &elapsed_time_ms);
            times.push_back(elapsed_time_ms);

            const int type = disparity_im.type();
            const uchar depth = type & CV_MAT_DEPTH_MASK;
            if (depth == CV_8U)
            {
                cv::Mat dispColor;
                cv::applyColorMap(disparity_im, dispColor, cv::COLORMAP_JET);
                cv::cvtColor(frame_0, frame_0, cv::COLOR_GRAY2BGR);
                cv::hconcat(dispColor, frame_0, dispColor);
                cv::resize(dispColor, dispColor, cv::Size(1920, 720));
                cv::imshow("disp8", dispColor);
                char key = (char)cv::waitKey(25);
                if (key == 27)
                    break;
            }
            else
            {
                cv::Mat disparity16(disparity_im.rows, disparity_im.cols, CV_16UC1);
                for (int i = 0; i < disparity_im.rows; i++)
                {
                    for (int j = 0; j < disparity_im.cols; j++)
                    {
                        const float d = disparity_im.at<float>(i, j) * 256.0f;
                        disparity16.at<uint16_t>(i, j) = (uint16_t)d;
                    }
                }
                cv::imshow("disp 16", disparity16);
                char key = (char)cv::waitKey(25);
                if (key == 27)
                    break;
            }

            if (times.size() % 100 == 0)
            {
                double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                std::cout << "It took an average of " << mean << " miliseconds, " << 1000.0f / mean << " fps" << std::endl;
                times.clear();
            }
        }
    }

    finish_disparity_method();

    return 0;
}
