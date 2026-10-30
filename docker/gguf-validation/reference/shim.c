/* Standalone harness; upstream Q4_0 routine is unmodified in q4_0.inc. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GGML_RESTRICT restrict
#define QK4_0 32
typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0;
_Static_assert(sizeof(block_q4_0)==18,"Q4_0 layout");
static float half(uint16_t h) {
    unsigned e=(h>>10)&31, m=h&1023;
    float v=e==0 ? ldexpf((float)m,-24) : e==31 ? (m ? NAN : INFINITY) : ldexpf((float)(1024+m),(int)e-25);
    return h&32768 ? -v : v;
}
#define GGML_FP16_TO_FP32 half
#include "q4_0.inc"
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    FILE *f=fopen(argv[1],"rb"); if(!f) return 2;
    block_q4_0 x[128]; size_t n=fread(x,1,sizeof(x),f);
    if(ferror(f)||fgetc(f)!=EOF||!n||n%18) { fclose(f); return 2; } fclose(f);
    /* The extracted routine uses native half storage: explicitly normalize LE. */
    for(size_t i=0;i<n/18;i++) { const uint8_t *p=(const uint8_t *)&x[i].d; uint16_t h=(uint16_t)p[0]|((uint16_t)p[1]<<8); x[i].d=h; }
    float y[4096]; dequantize_row_q4_0(x,y,(int64_t)(n/18*32));
    for(size_t i=0;i<n/18*32;i++) { if(!isfinite(y[i])) return 2; printf("%.9g\n",y[i]); }
    return 0;
}
