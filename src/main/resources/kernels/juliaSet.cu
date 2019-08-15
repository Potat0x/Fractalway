extern "C"
__global__ void juliaSet(double zoom, double posX, double posY, int maxIter, int width, int height, int* argb, double cRe, double cIm)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = y * width + x;

    if(x >= width || y >= height){
        return;
    }

    double zNextRe = 0;
    double zNextIm = 0;

    double pRe = (x - width/2.0)*zoom+posX;
    double pIm = (y - height/2.0)*zoom+posY;

    double zPrevRe = pRe;
    double zPrevIm = pIm;

    int i = 0;
    while(i++ < maxIter-1)
    {
        // zNext = zPrev*zPrev + c
        zNextRe = zPrevRe * zPrevRe - zPrevIm * zPrevIm + cRe;
        zNextIm = 2.0*zPrevRe*zPrevIm + cIm;

        // |zPrev| > 4.0
        if((zNextRe * zNextRe + zNextIm * zNextIm) > 4.0){
            break;
        }

        zPrevRe = zNextRe;
        zPrevIm = zNextIm;
    }

    double color = (255.0*i)/(1.0*maxIter);
    int r = 17.0*(abs(255-color)/255.0);
    int g = 255.0*(color/255.0);
    int b = 33.0*(abs(255-color)/255.0);
    argb[idx] = (255<<24) | (r<<16) | (g<<8) | b;
}
