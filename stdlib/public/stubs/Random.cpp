//===--- Random.cpp -------------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

// swift_stdlib_random
//
// Should the implementation of this function add a new platform/change for a
// platform, make sure to also update the documentation regarding platform
// implementation of this function.
// This can be found at: /docs/Random.md

#if defined(_WIN32) && !defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <Bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#else
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#if __has_include(<sys/random.h>)
#include <sys/random.h>
#endif
#include <sys/stat.h>
#if __has_include(<sys/syscall.h>)
#include <sys/syscall.h>
#endif

#include <stdlib.h>

#include "swift/Runtime/Debug.h"
#include "swift/Runtime/Mutex.h"
#include "../SwiftShims/Random.h"

// AZTODO: begin

#include <stddef.h>
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <limits>
#include <iostream>

#define SKEIN_KS_PARITY32         0x1BD11BDA
#ifndef THREEFRY2x32_DEFAULT_ROUNDS
#define THREEFRY2x32_DEFAULT_ROUNDS 20
#endif

#define R123_ULONG_LONG uint64_t
#define R123_BUILTIN_EXPECT(expr,likely) __builtin_expect(expr,likely)

enum r123_enum_threefry32x2 {
    /* Output from skein_rot_search (srs32x2-X5000.out)
    // Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
    // Start: Tue Jul 12 11:11:33 2011
    // rMin = 0.334. #0206[*07] [CRC=1D9765C0. hw_OR=32. cnt=16384. blkSize=  64].format   */
    R_32x2_0_0=13,
    R_32x2_1_0=15,
    R_32x2_2_0=26,
    R_32x2_3_0= 6,
    R_32x2_4_0=17,
    R_32x2_5_0=29,
    R_32x2_6_0=16,
    R_32x2_7_0=24

    /* 4 rounds: minHW =  4  [  4  4  4  4 ]
    // 5 rounds: minHW =  6  [  6  8  6  8 ]
    // 6 rounds: minHW =  9  [  9 12  9 12 ]
    // 7 rounds: minHW = 16  [ 16 24 16 24 ]
    // 8 rounds: minHW = 32  [ 32 32 32 32 ]
    // 9 rounds: minHW = 32  [ 32 32 32 32 ]
    //10 rounds: minHW = 32  [ 32 32 32 32 ]
    //11 rounds: minHW = 32  [ 32 32 32 32 ] */
    };

static inline uint32_t RotL_32(uint32_t x, unsigned int N)
{
    return (x << (N & 31)) | (x >> ((32-N) & 31));
}

template <typename value_type>
inline value_type assemble_from_u32(uint32_t *p32){
    value_type v=0;
    for(size_t i=0; i<(3+sizeof(value_type))/4; ++i)
        v |= ((value_type)(*p32++)) << (32*i);
    return v;
}

// Work-alike methods and typedefs modeled on std::array:
#define CXXMETHODS(_N, W, T)                                            \
    typedef T value_type;                                               \
    typedef T* iterator;                                                \
    typedef const T* const_iterator;                                    \
    typedef value_type& reference;                                      \
    typedef const value_type& const_reference;                          \
    typedef size_t size_type;                                           \
    typedef ptrdiff_t difference_type;                                  \
    typedef T* pointer;                                                 \
    typedef const T* const_pointer;                                     \
    typedef std::reverse_iterator<iterator> reverse_iterator;           \
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator; \
    /* Boost.array has static_size.  C++11 specializes tuple_size */    \
    enum {static_size = _N};                                            \
    reference operator[](size_type i){return v[i];}                     \
    const_reference operator[](size_type i) const {return v[i];}        \
    reference at(size_type i){ if(i >=  _N) assert("array index OOR"); return (*this)[i]; } \
    const_reference at(size_type i) const { if(i >=  _N) assert("array index OOR"); return (*this)[i]; } \
    size_type size() const { return  _N; }                              \
    size_type max_size() const { return _N; }                           \
    bool empty() const { return _N==0; };                               \
    iterator begin() { return &v[0]; }                                  \
    iterator end() { return &v[_N]; }                                   \
    const_iterator begin() const { return &v[0]; }                      \
    const_iterator end() const { return &v[_N]; }                       \
    const_iterator cbegin() const { return &v[0]; }                     \
    const_iterator cend() const { return &v[_N]; }                      \
    reverse_iterator rbegin(){ return reverse_iterator(end()); }        \
    const_reverse_iterator rbegin() const{ return const_reverse_iterator(end()); } \
    reverse_iterator rend(){ return reverse_iterator(begin()); }        \
    const_reverse_iterator rend() const{ return const_reverse_iterator(begin()); } \
    const_reverse_iterator crbegin() const{ return const_reverse_iterator(cend()); } \
    const_reverse_iterator crend() const{ return const_reverse_iterator(cbegin()); } \
    pointer data(){ return &v[0]; }                                     \
    const_pointer data() const{ return &v[0]; }                         \
    reference front(){ return v[0]; }                                   \
    const_reference front() const{ return v[0]; }                       \
    reference back(){ return v[_N-1]; }                                 \
    const_reference back() const{ return v[_N-1]; }                     \
    bool operator==(const r123array##_N##x##W& rhs) const{ \
  /* CUDA3 does not have std::equal */ \
  for (size_t i = 0; i < _N; ++i) \
      if (v[i] != rhs.v[i]) return false; \
  return true; \
    } \
    bool operator!=(const r123array##_N##x##W& rhs) const{ return !(*this == rhs); } \
    /* CUDA3 does not have std::fill_n */ \
    void fill(const value_type& val){ for (size_t i = 0; i < _N; ++i) v[i] = val; } \
    void swap(r123array##_N##x##W& rhs){ \
  /* CUDA3 does not have std::swap_ranges */ \
  for (size_t i = 0; i < _N; ++i) { \
      T tmp = v[i]; \
      v[i] = rhs.v[i]; \
      rhs.v[i] = tmp; \
  } \
    } \
    r123array##_N##x##W& incr(R123_ULONG_LONG n=1){                         \
        /* This test is tricky because we're trying to avoid spurious   \
           complaints about illegal shifts, yet still be compile-time   \
           evaulated. */                                                \
        if(sizeof(T)<sizeof(n) && n>>((sizeof(T)<sizeof(n))?8*sizeof(T):0) ) \
            return incr_carefully(n);                                   \
        if(n==1){                                                       \
            ++v[0];                                                     \
            if(_N==1 || R123_BUILTIN_EXPECT(!!v[0], 1)) return *this;   \
        }else{                                                          \
            v[0] += n;                                                  \
            if(_N==1 || R123_BUILTIN_EXPECT(n<=v[0], 1)) return *this;  \
        }                                                               \
        /* We expect that the N==?? tests will be                       \
           constant-folded/optimized away by the compiler, so only the  \
           overflow tests (!!v[i]) remain to be done at runtime.  For  \
           small values of N, it would be better to do this as an       \
           uncondtional sequence of adc.  An experiment/optimization    \
           for another day...                                           \
           N.B.  The weird subscripting: v[_N>3?3:0] is to silence      \
           a spurious error from icpc                                   \
           */                                                           \
        ++v[_N>1?1:0];                                                  \
        if(_N==2 || R123_BUILTIN_EXPECT(!!v[_N>1?1:0], 1)) return *this; \
        ++v[_N>2?2:0];                                                  \
        if(_N==3 || R123_BUILTIN_EXPECT(!!v[_N>2?2:0], 1)) return *this;  \
        ++v[_N>3?3:0];                                                  \
        for(size_t i=4; i<_N; ++i){                                     \
            if( R123_BUILTIN_EXPECT(!!v[i-1], 1) ) return *this;        \
            ++v[i];                                                     \
        }                                                               \
        return *this;                                                   \
    }                                                                   \
    /* seed(SeedSeq) would be a constructor if having a constructor */  \
    /* didn't cause headaches with defaults */                          \
    template <typename SeedSeq>                                         \
    static r123array##_N##x##W seed(SeedSeq &ss){      \
        r123array##_N##x##W ret;                                        \
        const size_t Ngen = _N*((3+sizeof(value_type))/4);              \
        uint32_t u32[Ngen];                                             \
        uint32_t *p32 = &u32[0];                                        \
        ss.generate(&u32[0], &u32[Ngen]);                               \
        for(size_t i=0; i<_N; ++i){                                     \
            ret.v[i] = assemble_from_u32<value_type>(p32);              \
            p32 += (3+sizeof(value_type))/4;                            \
        }                                                               \
        return ret;                                                     \
    }                                                                   \
protected:                                                              \
    r123array##_N##x##W& incr_carefully(R123_ULONG_LONG n){ \
        /* n may be greater than the maximum value of a single value_type */ \
        value_type vtn;                                                 \
        vtn = n;                                                        \
        v[0] += n;                                                      \
        const unsigned rshift = 8* ((sizeof(n)>sizeof(value_type))? sizeof(value_type) : 0); \
        for(size_t i=1; i<_N; ++i){                                     \
            if(rshift){                                                 \
                n >>= rshift;                                           \
            }else{                                                      \
                n=0;                                                    \
            }                                                           \
            if( v[i-1] < vtn )                                          \
                ++n;                                                    \
            if( n==0 ) break;                                           \
            vtn = n;                                                    \
            v[i] += n;                                                  \
        }                                                               \
        return *this;                                                   \
    }                                                                   \

template<typename T>
struct r123arrayinsertable{
    const T& v;
    r123arrayinsertable(const T& t_) : v(t_) {} 
    friend std::ostream& operator<<(std::ostream& os, const r123arrayinsertable<T>& t){
        return os << t.v;
    }
};

template<>
struct r123arrayinsertable<uint8_t>{
    const uint8_t& v;
    r123arrayinsertable(const uint8_t& t_) : v(t_) {} 
    friend std::ostream& operator<<(std::ostream& os, const r123arrayinsertable<uint8_t>& t){
        return os << (int)t.v;
    }
};

template<typename T>
struct r123arrayextractable{
    T& v;
    r123arrayextractable(T& t_) : v(t_) {}
    friend std::istream& operator>>(std::istream& is, r123arrayextractable<T>& t){
        return is >> t.v;
    }
};

template<>
struct r123arrayextractable<uint8_t>{
    uint8_t& v;
    r123arrayextractable(uint8_t& t_) : v(t_) {} 
    friend std::istream& operator>>(std::istream& is, r123arrayextractable<uint8_t>& t){
        int i;
        is >>  i;
        t.v = i;
        return is;
    }
};


#define CXXOVERLOADS(_N, W, T)                                          \
                                                                        \
inline std::ostream& operator<<(std::ostream& os, const r123array##_N##x##W& a){   \
    os << r123arrayinsertable<T>(a.v[0]);                                  \
    for(size_t i=1; i<_N; ++i)                                          \
        os << " " << r123arrayinsertable<T>(a.v[i]);                       \
    return os;                                                          \
}                                                                       \
                                                                        \
inline std::istream& operator>>(std::istream& is, r123array##_N##x##W& a){         \
    for(size_t i=0; i<_N; ++i){                                         \
        r123arrayextractable<T> x(a.v[i]);                                 \
        is >> x;                                                        \
    }                                                                   \
    return is;                                                          \
}                                                                       \
                                                                        \
namespace r123{                                                        \
 typedef r123array##_N##x##W Array##_N##x##W;                          \
}

#define _r123array_tpl(_N, W, T)                   \
    /** @ingroup arrayNxW */                        \
    /** @see arrayNxW */                            \
struct r123array##_N##x##W{                         \
 T v[_N];                                       \
 CXXMETHODS(_N, W, T)                           \
};                                              \
                                                \
CXXOVERLOADS(_N, W, T)

/** @endcond */

_r123array_tpl(2, 32, uint32_t)  /* r123array2x32 */

#define _threefry2x_tpl(W)                                              \
typedef struct r123array2x##W threefry2x##W##_ctr_t;                          \
typedef struct r123array2x##W threefry2x##W##_key_t;                          \
typedef struct r123array2x##W threefry2x##W##_ukey_t;                          \
static inline threefry2x##W##_key_t threefry2x##W##keyinit(threefry2x##W##_ukey_t uk) { return uk; } \
static inline threefry2x##W##_ctr_t threefry2x##W##_R(unsigned int Nrounds, threefry2x##W##_ctr_t in, threefry2x##W##_key_t k) __attribute__((always_inline)); \
static inline                                          \
threefry2x##W##_ctr_t threefry2x##W##_R(unsigned int Nrounds, threefry2x##W##_ctr_t in, threefry2x##W##_key_t k){ \
    threefry2x##W##_ctr_t X;                                              \
    uint##W##_t ks[2+1];                                          \
    int  i; /* avoid size_t to avoid need for stddef.h */                   \
    /*R123_ASSERT(Nrounds<=32); */                                          \
    ks[2] =  SKEIN_KS_PARITY##W;                                   \
    for (i=0;i < 2; i++)                                        \
        {                                                               \
            ks[i] = k.v[i];                                             \
            X.v[i]  = in.v[i];                                          \
            ks[2] ^= k.v[i];                                    \
        }                                                               \
                                                                        \
    /* Insert initial key before round 0 */                             \
    X.v[0] += ks[0]; X.v[1] += ks[1];                                   \
                                                                        \
    if(Nrounds>0){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_0_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>1){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_1_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>2){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_2_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>3){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_3_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>3){                                                      \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2];                               \
        X.v[1] += 1;     /* X.v[2-1] += r  */                   \
    }                                                                   \
    if(Nrounds>4){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_4_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>5){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_5_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>6){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_6_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>7){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_7_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>7){                                                      \
        /* InjectKey(r=2) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[0];                               \
        X.v[1] += 2;                                                    \
    }                                                                   \
    if(Nrounds>8){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_0_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>9){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_1_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>10){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_2_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>11){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_3_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>11){                                                     \
        /* InjectKey(r=3) */                                            \
        X.v[0] += ks[0]; X.v[1] += ks[1];                               \
        X.v[1] += 3;                                                    \
    }                                                                   \
    if(Nrounds>12){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_4_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>13){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_5_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>14){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_6_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>15){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_7_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>15){                                                     \
        /* InjectKey(r=4) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2];                               \
        X.v[1] += 4;                                                    \
    }                                                                   \
    if(Nrounds>16){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_0_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>17){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_1_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>18){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_2_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>19){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_3_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>19){                                                     \
        /* InjectKey(r=5) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[0];                               \
        X.v[1] += 5;                                                    \
    }                                                                   \
    if(Nrounds>20){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_4_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>21){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_5_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>22){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_6_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>23){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_7_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>23){                                                     \
        /* InjectKey(r=6) */                                            \
        X.v[0] += ks[0]; X.v[1] += ks[1];                               \
        X.v[1] += 6;                                                    \
    }                                                                   \
    if(Nrounds>24){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_0_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>25){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_1_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>26){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_2_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>27){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_3_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>27){                                                     \
        /* InjectKey(r=7) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2];                               \
        X.v[1] += 7;                                                    \
    }                                                                   \
    if(Nrounds>28){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_4_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>29){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_5_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>30){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_6_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>31){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_7_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>31){                                                     \
        /* InjectKey(r=8) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[0];                               \
        X.v[1] += 8;                                                    \
    }                                                                   \
    return X;                                                           \
}                                                                       \
 /** @ingroup ThreefryNxW */                                            \
enum r123_enum_threefry2x##W { threefry2x##W##_rounds = THREEFRY2x##W##_DEFAULT_ROUNDS };       \
static inline threefry2x##W##_ctr_t threefry2x##W(threefry2x##W##_ctr_t in, threefry2x##W##_key_t k) __attribute__((always_inline)); \
static inline                                     \
threefry2x##W##_ctr_t threefry2x##W(threefry2x##W##_ctr_t in, threefry2x##W##_key_t k){ \
    return threefry2x##W##_R(threefry2x##W##_rounds, in, k);            \
}

_threefry2x_tpl(32)

#define threefry2x32(c,k) threefry2x32_R(threefry2x32_rounds, c, k)

// AZTODO: end

//#if defined(__APPLE__)

SWIFT_RUNTIME_STDLIB_API
void swift::swift_stdlib_random(void *buf, __swift_size_t nbytes) {
  thread_local uint64_t ctr = 0;
  thread_local threefry2x32_key_t key = {{
    (uint32_t)arc4random(),
    (uint32_t)arc4random()
  }};

  size_t byte_count = 0;
  size_t slices_remaining = 0;
  while (true) {
    // Build the 2-ple
    uint8_t r0, r1, r2, r3;
    uint8_t r4, r5, r6, r7;

    if (slices_remaining == 0) {
      threefry2x32_ctr_t ctr_arr = {{
        (uint32_t)((ctr & 0xFFFFFFFF00000000) >> 32),
        (uint32_t)((ctr & 0x00000000FFFFFFFF))
      }};
      threefry2x32_ctr_t rand = threefry2x32(ctr_arr, key);

      slices_remaining = 8;

      r0 = (rand.v[0] & 0x000000FF);
      r1 = (rand.v[0] & 0x0000FF00) >> 8;
      r2 = (rand.v[0] & 0x00FF0000) >> 16;
      r3 = (rand.v[0] & 0xFF000000) >> 24;

      r4 = (rand.v[1] & 0x000000FF);
      r5 = (rand.v[1] & 0x0000FF00) >> 8;
      r6 = (rand.v[1] & 0x00FF0000) >> 16;
      r7 = (rand.v[1] & 0xFF000000) >> 24;
    }

    switch (slices_remaining) {
      case 8:
        *((uint8_t*)buf + byte_count) = r0;
        break;
      case 7:
        *((uint8_t*)buf + byte_count) = r1;
        break;
      case 6:
        *((uint8_t*)buf + byte_count) = r2;
        break;
      case 5:
        *((uint8_t*)buf + byte_count) = r3;
        break;
      case 4:
        *((uint8_t*)buf + byte_count) = r4;
        break;
      case 3:
        *((uint8_t*)buf + byte_count) = r5;
        break;
      case 2:
        *((uint8_t*)buf + byte_count) = r6;
        break;
      case 1:
        *((uint8_t*)buf + byte_count) = r7;
        break;
      default:
        assert(false);
    }
    slices_remaining -= 1;

    // Increment the counter
    ctr += 1;
    // Increment the byte_count
    byte_count += 1;
    if (byte_count >= nbytes) {
      return;
    }
  }

  // arc4random_buf(buf, nbytes);
}
/*
#elif defined(_WIN32) && !defined(__CYGWIN__)
#warning TODO: Test swift_stdlib_random on Windows

SWIFT_RUNTIME_STDLIB_API
void swift::swift_stdlib_random(void *buf, __swift_size_t nbytes) {
  NTSTATUS status = BCryptGenRandom(nullptr,
                                    static_cast<PUCHAR>(buf),
                                    static_cast<ULONG>(nbytes),
                                    BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  if (!BCRYPT_SUCCESS(status)) {
    fatalError(0, "Fatal error: 0x%.8X in '%s'\n", status, __func__);
  }
}

#else

#undef  WHILE_EINTR
#define WHILE_EINTR(expression) ({                                             \
  decltype(expression) result = -1;                                            \
  do { result = (expression); } while (result == -1 && errno == EINTR);        \
  result;                                                                      \
})

SWIFT_RUNTIME_STDLIB_API
void swift::swift_stdlib_random(void *buf, __swift_size_t nbytes) {
  while (nbytes > 0) {
    __swift_ssize_t actual_nbytes = -1;

#if defined(__NR_getrandom)
    static const bool getrandom_available =
      !(syscall(__NR_getrandom, nullptr, 0, 0) == -1 && errno == ENOSYS);
  
    if (getrandom_available) {
      actual_nbytes = WHILE_EINTR(syscall(__NR_getrandom, buf, nbytes, 0));
    }
#elif __has_include(<sys/random.h>) && (defined(__CYGWIN__) || defined(__Fuchsia__))
    __swift_size_t getentropy_nbytes = std::min(nbytes, __swift_size_t{256});
    
    if (0 == getentropy(buf, getentropy_nbytes)) {
      actual_nbytes = getentropy_nbytes;
    }
#endif

    if (actual_nbytes == -1) {
      static const int fd = 
        WHILE_EINTR(open("/dev/urandom", O_RDONLY | O_CLOEXEC, 0));
        
      if (fd != -1) {
        static StaticMutex mutex;
        mutex.withLock([&] {
          actual_nbytes = WHILE_EINTR(read(fd, buf, nbytes));
        });
      }
    }
    
    if (actual_nbytes == -1) {
      fatalError(0, "Fatal error: %d in '%s'\n", errno, __func__);
    }
    
    buf = static_cast<uint8_t *>(buf) + actual_nbytes;
    nbytes -= actual_nbytes;
  }
}

#endif
*/
