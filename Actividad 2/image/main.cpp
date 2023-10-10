#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>

int main() {
    std::string image_path = "C:\\Users\\xxdar\\Pictures\\mrbeast.jpg";
    std::cout << "Enter the path to the image" << std::endl;
    std::cin >> image_path;

    if(!std::filesystem::exists(image_path)){
        std::cout << "File does not exist at the specified path" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(image_path);

    if(image.empty()){
        std::cout << "Error loading the image" << std::endl;
        return -1;
    }
    else {
        std::cout << "Image loaded successfully" << std::endl;
    }

    cv::imshow("Image", image);

    cv::waitKey(0);

    cv::Mat bgr[3];
    cv::split(image, bgr);

    cv::Mat blueChannel, greenChannel, redChannel;

    cv::applyColorMap(bgr[0], blueChannel, cv::COLORMAP_JET);
    cv::applyColorMap(bgr[1], greenChannel, cv::COLORMAP_SPRING);
    cv::applyColorMap(bgr[2], redChannel, cv::COLORMAP_HOT);

    cv::imshow("Blue Channel", blueChannel);
    cv::imshow("Green Channel", greenChannel);
    cv::imshow("Red Channel", redChannel);

    cv::waitKey(0);

    cv::applyColorMap(bgr[0], blueChannel, cv::COLORMAP_BONE);
    cv::applyColorMap(bgr[1], greenChannel, cv::COLORMAP_BONE);
    cv::applyColorMap(bgr[2], redChannel, cv::COLORMAP_BONE);

    cv::imshow("Blue Channel", blueChannel);
    cv::imshow("Green Channel", greenChannel);
    cv::imshow("Red Channel", redChannel);

    cv::waitKey(0);
    return 0;
}