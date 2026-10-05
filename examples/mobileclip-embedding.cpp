#include "photon/mobileclip.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static std::vector<float> read_f32(const std::string& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot open: "+p);f.seekg(0,std::ios::end);size_t n=(size_t)f.tellg();if(n!=3ull*256ull*256ull*sizeof(float))throw std::runtime_error("Expected 3*256*256 float32 values");f.seekg(0);std::vector<float>x(n/sizeof(float));f.read((char*)x.data(),(std::streamsize)n);return x;}
int main(int argc,char**argv){try{if(argc!=3){std::cerr<<"Usage: mobileclip-embedding <mobileclip-s1.onnx> <image_chw.f32>\n";return 2;}photon::MobileCLIPS1 m;m.load(argv[1]);photon::ImageTensor im;im.channels=3;im.height=256;im.width=256;im.data=read_f32(argv[2]);auto e=m.encode(im,true);std::cout<<"backend=onnxruntime-cpu\nembedding_dim="<<e.size()<<"\nembedding[0:8]=";for(size_t i=0;i<8;++i)std::cout<<(i?',':'\0')<<e[i];std::cout<<"\n";std::ofstream o("mobileclip_embedding.f32",std::ios::binary);o.write((char*)e.data(),(std::streamsize)e.size()*sizeof(float));return 0;}catch(const std::exception&e){std::cerr<<"mobileclip-embedding: "<<e.what()<<'\n';return 1;}}
