extern "C"
__global__ void burningShip(double zoom, double posX, double posY, int maxIter, int width, int height, int* argb)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = y * width + x;

    if(x >= width || y >= height){
        return;
    }

    double zPrevRe = 0;
    double zPrevIm = 0;
    double zNextRe = 0;
    double zNextIm = 0;

    double pRe = (x - width/2.0)*zoom+posX;
    double pIm = (y - height/2.0)*zoom+posY;

    int i = 0;
    while(i++ < maxIter-1)
    {
        // zNext = zPrev*zPrev + p
        zNextRe = zPrevRe * zPrevRe - zPrevIm * zPrevIm + pRe;
        zNextIm = 2.0*zPrevRe*zPrevIm + pIm;

        // |zPrev| > 4.0
        if((zNextRe * zNextRe + zNextIm * zNextIm) > 4.0){
            break;
        }

        zPrevRe = fabs(zNextRe);
        zPrevIm = fabs(zNextIm);
    }

    int color = (255.0*i)/maxIter;
    argb[idx] = (255<<24) | (color<<16) | (color<<8) | color;
}
