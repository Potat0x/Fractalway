extern "C"
__global__ void mandelbrotSet(double zoom, double posX, double posY, int maxIter, int* r, int* g, int* b)
{
    const int wh = 512;

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = y * wh + x;

    if(x >= wh || y >= wh){
        return;
    }

    double zPrevRe = 0;
    double zPrevIm = 0;
    double zNextRe = 0;
    double zNextIm = 0;

    double pRe = (x - wh/2.0)*zoom+posX;
    double pIm = (y - wh/2.0)*zoom+posY;

    int i = 0;
    while(i++ < maxIter)
    {
        // zNext = zPrev*zPrev + p
        zNextRe = zPrevRe * zPrevRe - zPrevIm * zPrevIm + pRe;
        zNextIm = 2.0*zPrevRe*zPrevIm + pIm;

        // |zPrev| > 4.0
        if((zNextRe * zNextRe + zNextIm * zNextIm) > 4.0){
            break;
        }

        zPrevRe = zNextRe;
        zPrevIm = zNextIm;
    }

    double color = (255.0*i)/(1.0*maxIter);
//    r[idx] = 0.45*color;
//    g[idx] = color;
//    b[idx] = abs(255-color)/3.0;
    r[idx] = 0.55*color;
    g[idx] = color*0.9;
    b[idx] = abs(255-color)/3.0;
}
