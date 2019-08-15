extern "C"
__global__ void gradient(double zoom, double posX, double posY, int maxIter, int width, int height, int* argb)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = y * width + x;

    if(x >= width || y >= height){
        return;
    }

    float centerDist = hypotf(width/2 - x, height/2 - y);
    if(centerDist > 255){
        centerDist = 255.0;
    }

    double rgbScale = 255.0/width;

    if(threadIdx.x == 0 || threadIdx.y == 0){
        rgbScale *= 0.8;
    }

    int r = x*rgbScale;
    int g = y*rgbScale;
    int b = centerDist*rgbScale*0.5;
    argb[idx] = (255<<24) | (r<<16) | (g<<8) | b;
}
