
#include "Image/ImgIO.h"
#include "Image/ImgProcessGPU.h"
#include <boost/filesystem.hpp>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
using std::cout;
using std::endl;

using std::vector;
using namespace boost::filesystem;

struct recursive_directory_range
{
    typedef recursive_directory_iterator iterator;
    recursive_directory_range(path p) : p_(p) {}

    iterator begin() { return recursive_directory_iterator(p_); }
    iterator end() { return recursive_directory_iterator(); }

    path p_;
};

int main(int argc, char *argv[])
{
    // // std::string imgs_dir = std::string("../../data/clark/");
    // // std::string weis_dir = std::string("../../data/clark_segnet_mask/");

    // vector<std::string> img_paths;
    // vector<std::string> wei_paths;

    // for (auto it : recursive_directory_range(argv[1]))
    // {
    //     img_paths.push_back(std::string(it.path().c_str()));  
    // }

    // for (auto it : recursive_directory_range(argv[2]))
    // {
    //     wei_paths.push_back(std::string(it.path().c_str()));  
    // }

    // std::string out_dir(argv[3]);
    // if (out_dir.back() == '/')
    // {
    //     out_dir.pop_back();
    // }

    // ImgIO io;
    // ImgProcessGPU proc;


    // for (int i = 0; i < wei_paths.size(); ++i)
    // //for (int i = 41; i < 42; ++i)
    // {
    //     // printf("%s\n", img_paths[i].c_str());
    //     // printf("%s\n", wei_paths[i].c_str());

    //     BaseImgGPU<float> img, wei, out;
    //     io.loadImageGPU(img_paths[i].c_str(), img);
    //     io.loadImageGPU(wei_paths[i].c_str(), wei);
    //     double radius_factor = 0.02;

    //     //cout << wei.channel() << endl;

    //     for (int k = 0; k < 4; ++k)
    //     {
    //         radius_factor /= 2.0;
    //         int radius = radius_factor * wei.width();
    //         proc.jointBilateralFilterFloat(img, wei, out, radius, 0.005, 10.f/255 * 10.f/255, 0.25 * radius * radius);
    //         wei.copy_from(&out);
    //         proc.jointBilateralFilterFloat(img, wei, out, radius, 0.005, 10.f/255 * 10.f/255, 0.25 * radius * radius);
    //         wei.copy_from(&out);
    //     }

    //     cout << i << endl;

    //     char out_path[255];
    //     sprintf(out_path, "%s/clark.%04d.png", out_dir.c_str(), i+1);
    //     io.saveImageGPU(out_path, out);
    // }



    ImgIO io;
    ImgProcessGPU proc;

    BaseImgGPU<float> im1, im2, im3;
    io.loadImageGPU("../../data/clark/clark.0001.png",im1);

    proc.resizeImageFloat(im1,im2,2*im1.width(),2*im1.height(),1.f);

    cv::Mat cvImg;
    io.base_to_cv(im2, cvImg);
    io.cv_to_base(cvImg, im3);
    io.saveImageGPU("resized.jpg", im3);


    cv::namedWindow( "resize image", CV_WINDOW_AUTOSIZE );

    cv::imshow( "resize image", cvImg );

    cv::waitKey(0);

    return 0;
}
