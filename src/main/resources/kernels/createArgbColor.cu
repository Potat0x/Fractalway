__device__ int createArgbColor(int iter, int maxIter)
{
    int color = (255.0*iter)/maxIter;
    return(255<<24) | (color<<16) | (color<<8) | color;
}
