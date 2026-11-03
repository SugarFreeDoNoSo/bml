/* Independent finite-real BML diagnostic, not a production inference kernel.
 * Build flags are fixed by bounded_inference_oracle.py; fast-math is disabled.
 * Native multiplication/addition appear only in the separate reference projector
 * and in configuration constants, bounds, counters and shape/index calculations.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint64_t nodes;
    double max_exponent, min_argument, max_abs_value;
    int error;
    double bad_left, bad_right;
} Stats;
static Stats state;
void reset_stats(void) {
    state = (Stats){.max_exponent=-INFINITY,.min_argument=INFINITY};
}
Stats read_stats(void) { return state; }
static double B(double a, double b) {
    state.nodes++;
    if (state.error) return NAN;
    if (!isfinite(a) || !isfinite(b) || b <= 0.0) {
        state.error=1; state.bad_left=a; state.bad_right=b; return NAN;
    }
    double ex=exp2(a);
    if (!isfinite(ex) || ex == 0.0) {
        state.error=2; state.bad_left=a; state.bad_right=b; return NAN;
    }
    double out=ex-log2(b);
    if (!isfinite(out)) {
        state.error=3; state.bad_left=a; state.bad_right=b; return NAN;
    }
    if(a>state.max_exponent) state.max_exponent=a;
    if(b<state.min_argument) state.min_argument=b;
    if(fabs(out)>state.max_abs_value) state.max_abs_value=fabs(out);
    return out;
}
static double E(double x) { return B(x,1.0); }
static double L(double x) { return B(1.0,B(B(1.0,x),1.0)); }
static double D(double a,double b) { return B(L(a),E(b)); }
static double A(double a,double b,double c) { return D(c,D(D(c,a),b)); }
static double N(double a,double c) { return D(D(c,a),c); }
static double PA(double a,double b,double c) { return D(a,N(b,c)); }
static double P(double a,double b) { return E(A(L(a),L(b),64.0)); }
static double OA(double a,double b) {
    double p=E(a),q=B(a,p);return D(p,D(q,b));
}
static double OP(double a,double b) { return E(OA(L(a),L(b))); }
static double OS(double a,double b) {
    double p=E(a),q=B(a,p),r=E(b),s=B(b,r);
    return D(OA(OP(p,r),OP(q,s)),OA(OP(p,s),OP(q,r)));
}
static double U(double x,double m) { return D(P(A(x,2*m,2*m),1/m),2.0); }
static double S(double x,double y,double m,double n) {
    double t=OS(U(x,m),U(y,n));return D(P(A(t,2.0,2.0),m*n),2*m*n);
}
static double R(double x) { return E(N(L(x),64.0)); }
double bounded_product(double x,double y,double m,double n) { return S(x,y,m,n); }
double bounded_add(double x,double y) { return A(x,y,64.0); }
double bounded_scale(double x,double y) { return S(x,y,256.0,1.0); }
double bounded_swiglu(double g,double u) {
    double shift=A(g,-16.0,32.0);
    double num=E(S(shift,log2(exp(1.0)),32.0,2.0));
    double den=PA(exp(-16.0),num,2.0);
    double sig=P(num,R(den));
    return S(S(g,sig,16.0,1.0),u,16.0,16.0);
}
void bounded_rope(double x,double y,double c,double s,double* out) {
    out[0]=A(S(x,c,16.0,1.0),N(S(y,s,16.0,1.0),32.0),32.0);
    out[1]=A(S(x,s,16.0,1.0),S(y,c,16.0,1.0),32.0);
}
int bounded_rms(const double*x,const float*w,size_t n,double eps,double*out) {
    const double m=16.0,wb=16.0,k=ceil(sqrt((double)n))+1.0;
    double total=(double)n*eps/(m*m);
    for(size_t i=0;i<n;i++) { double u=U(x[i],m);total=PA(total,OS(u,u),2.0); }
    double var=P(total,1.0/(double)n);
    double lb=fmax(fabs(log2(eps/(m*m))),fabs(log2(1+eps/(m*m))))+1.0;
    double root=E(S(L(var),0.5,lb,1.0)),inv=R(root),shift=P(k,root);
    for(size_t i=0;i<n;i++) {
        double norm=D(P(A(U(x[i],m),shift,2.0),inv),k);
        out[i]=S(norm,(double)w[i],k,wb);
        if(state.error) return (int)i+1;
    }
    return 0;
}
int bounded_dot(const double*x,const double*w,size_t n,double m,double wb,double*out) {
    double sum=0;
    for(size_t i=0;i<n;i++) {
        sum=A(sum,S(x[i],w[i],m,wb),256.0);
        if(state.error) return (int)i+1;
    }
    *out=sum;return 0;
}
int bounded_project(const double*x,const float*w,size_t width,size_t rows,double*out,double*max_partial) {
    *max_partial=0;
    for(size_t row=0;row<rows;row++) {
        double sum=0;
        for(size_t col=0;col<width;col++) {
            sum=A(sum,S(x[col],(double)w[row*width+col],128.0,2.0),256.0);
            if(state.error) return (int)row+1;
            *max_partial=fmax(*max_partial,fabs(sum));
        }
        out[row]=sum;
    }
    return 0;
}
int bounded_softmax(const double*x,size_t n,double*out) {
    double total=0,maximum=x[0];
    for(size_t i=1;i<n;i++) if(x[i]>maximum) maximum=x[i];
    double negative_maximum=N(maximum,32.0);
    for(size_t i=0;i<n;i++) {
        out[i]=E(S(A(x[i],negative_maximum,32.0),log2(exp(1.0)),32.0,2.0));
        total=i==0?out[i]:PA(total,out[i],2.0);
    }
    double inverse=R(total);
    for(size_t i=0;i<n;i++) out[i]=P(out[i],inverse);
    return state.error;
}
void native_project(const double*x,const float*w,size_t width,size_t rows,double*out,double*max_partial) {
    *max_partial=0;
    for(size_t row=0;row<rows;row++) {
        double sum=0;
        for(size_t col=0;col<width;col++) {
            sum+=x[col]*(double)w[row*width+col];
            *max_partial=fmax(*max_partial,fabs(sum));
        }
        out[row]=sum;
    }
}
