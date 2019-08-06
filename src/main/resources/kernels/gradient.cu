extern "C"
__global__ void gradient(double zoom, double startX, double startY, int maxIter, int* r, int* g, int* b)
{
    const int wh = 512;

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= wh || y >= wh){
        return;
    }

    float centerDist = hypotf(wh/2 - x, wh/2 - y);
    if(centerDist > 255){
        centerDist = 255.0;
    }

    double rgbScale = 255.0/wh;

    if(threadIdx.x == 0 || threadIdx.y == 0){
        rgbScale *= 0.8;
    }

    r[y*wh+x] = x*rgbScale;
    g[y*wh+x] = y*rgbScale;
    b[y*wh+x] = centerDist*rgbScale*0.5;
}
