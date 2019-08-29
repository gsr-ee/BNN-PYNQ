//
// Created by gsr on 17/08/2019.
//
#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <iostream>

using namespace std;
#define CIFAR10_IMAGE_DEPTH (3)
#define CIFAR10_IMAGE_WIDTH (32)
#define CIFAR10_IMAGE_HEIGHT (32)
#define CIFAR10_IMAGE_AREA (CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT)
#define CIFAR10_IMAGE_SIZE (CIFAR10_IMAGE_AREA*CIFAR10_IMAGE_DEPTH)

inline void parse_cifar(const std::string& filename,
                          std::vector<std::vector<float>> *train_images,
                          float scale_min,
                          float scale_max) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (ifs.fail() || ifs.bad())
        throw "Failed to open file %s" + filename;

    std::vector<unsigned char> buf(CIFAR10_IMAGE_SIZE);

    while (!ifs.eof()) {
        std::vector<float> img;

        if (!ifs.read((char*) &buf[0], CIFAR10_IMAGE_SIZE))
            break;

        std::transform(buf.begin(), buf.end(), std::back_inserter(img),
                [=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });


        train_images->push_back(img);
    }
}

#define MNIST_IMAGE_WIDTH (28)
#define MNIST_IMAGE_HEIGHT (28)
#define MNIST_IMAGE_SIZE (CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT)

inline void parse_mnist(const std::string& filename,
                 std::vector<std::vector<float>> *train_images,
                 float scale_min,
                 float scale_max){
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.is_open())
        throw "Failed to open file %s" + filename;

    std::vector<unsigned char> buf(MNIST_IMAGE_SIZE);

    //read the offsets and delete them since they are unused
    ifs.read((char*) &buf[0], 16);
    //buf.clear();
    //delete[] (char *) header[0];

    while (!ifs.eof()) {
        std::vector<float> img;

        if (!ifs.read((char*) &buf[0], MNIST_IMAGE_SIZE))
            break;

        std::transform(buf.begin(), buf.end(), std::back_inserter(img),
                       [=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });


        train_images->push_back(img);
    }
}


class Interleaver{
public:
    Interleaver(const unsigned int& pixelsPerChan, const unsigned int channels, const bool& deinterleave){
        this->channels_=channels;
        this->pixelsPerChan_=pixelsPerChan;
        this->deinterleave_=deinterleave;
    }

    void interleave(const vector<float>& in, vector<float>& out) {

        //float* out=new float[];
        out.resize(in.size());
        //out=new float[in.size()];

        for (unsigned int c = 0; c < channels_; c++) {
            for (unsigned int pix = 0; pix < pixelsPerChan_; pix++) {
                if (deinterleave_) {
                    out[c * pixelsPerChan_ + pix] = in[pix * channels_ + c];
                } else {
                    out[pix * channels_ + c] = in[c * pixelsPerChan_ + pix];
                }
            }
        }
    }

private:
    unsigned int channels_, pixelsPerChan_;
    bool deinterleave_;
};

#endif

/*
int main(){
    std::vector<std::vector<float>> train_images;

    parse_cifar10("/home/gsr/BNN-PYNQ/tests/Test_image/deer.bin", &train_images, 0, 255);

    for(unsigned int i =0; i<10; i++)
        std::cout << train_images[0][i] << std::endl;

}
*/
