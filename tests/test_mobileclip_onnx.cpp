#include "photon/mobileclip.h"
#include <cassert>
#include <iostream>
int main(){photon::MobileCLIPS1 m;const auto& c=m.config();assert(c.image_size==256);assert(c.embedding_dim==512);assert(c.stage0_blocks==4&&c.stage1_blocks==12&&c.stage2_blocks==20&&c.stage3_blocks==4);std::cout<<"MobileCLIP ONNX API test passed\n";}
