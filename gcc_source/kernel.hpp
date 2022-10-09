#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <cmath>

#pragma GCC diagnostic ignored "-Wpsabi"


typedef float Vec16f __attribute__((__vector_size__(64), __aligned__(64)));
typedef int Vec16i __attribute__((__vector_size__(64), __aligned__(64)));


template <typename T>
static T square(T x) {
    return x * x;
}

#define DK(name, value) const float name = static_cast<float>(value)
#define FMA(a, b, c) (a * b + c)
#define FMS(a, b, c) (a * b - c)
#define FNMS(a, b, c) (c - a * b)

#ifdef _MSC_VER
#pragma warning(disable: 4068)
#endif

static inline constexpr Vec16f __attribute__((__always_inline__)) constant(float val) {
    Vec16f ret = { val, val, val, val, val, val, val, val, val, val, val, val, val, val, val };
    return ret;
}

static inline Vec16f sqrt(Vec16f x) {
    Vec16f ret;

    for (int i = 0; i < 16; i++) {
        ret[i] = std::sqrt(x[i]);
    }

    return ret;
}

static inline Vec16f pow(Vec16f base, Vec16f exp) {
    Vec16f ret;

    for (int i = 0; i < 16; i++) {
        ret[i] = std::pow(base[i], exp[i]);
    }

    return ret;
}

static inline Vec16f max(Vec16f x, Vec16f y) {
#if __clang__ && __clang_major__ >= 14
    return __builtin_elementwise_max(x, y);
#else // __clang__ && __clang_major__ >= 14
    Vec16f ret;

    for (int i = 0; i < 16; i++) {
        ret[i] = std::fmax(x[i], y[i]);
    }

    return ret;
#endif // __clang__ && __clang_major__ >= 14
}

template <int n>
static void rdft(Vec16f data[(n / 2 + 1) * 2]);

// ./gen_r2cf.native -standalone -with-rs 2 -with-csr 2 -with-csi 2 -fma -n 16
template <>
void rdft<16>(Vec16f data[18]) {
    using E = Vec16f;

    auto R0 = &data[0];
    auto R1 = &data[1];
    auto Cr = R0;
    auto Ci = R1;

    DK(KP923879532, +0.923879532511286756128183189396788286822416626);
    DK(KP707106781, +0.707106781186547524400844362104849039284835938);
    DK(KP414213562, +0.414213562373095048801688724209698078569671875);

    E T3;
    E T6;
    E T7;
    E T49;
    E T37;
    E T10;
    E T13;
    E T14;
    E T50;
    E T40;
    E T22;
    E T55;
    E T34;
    E T44;
    E T29;
    E T54;
    E T35;
    E T47;
    E T38;
    E T39;
    E T53;
    E T56;
    {
        E T1;
        E T2;
        E T4;
        E T5;
        T1 = R0[0];
        T2 = R0[8];
        T3 = T1 + T2;
        T4 = R0[4];
        T5 = R0[12];
        T6 = T4 + T5;
        T7 = T3 + T6;
        T49 = T4 - T5;
        T37 = T1 - T2;
    }
    {
        E T8;
        E T9;
        E T11;
        E T12;
        T8 = R0[2];
        T9 = R0[10];
        T10 = T8 + T9;
        T38 = T8 - T9;
        T11 = R0[14];
        T12 = R0[6];
        T13 = T11 + T12;
        T39 = T11 - T12;
    }
    T14 = T10 + T13;
    T50 = T39 - T38;
    T40 = T38 + T39;
    {
        E T18;
        E T42;
        E T21;
        E T43;
        {
            E T16;
            E T17;
            E T19;
            E T20;
            T16 = R1[0];
            T17 = R1[8];
            T18 = T16 + T17;
            T42 = T16 - T17;
            T19 = R1[4];
            T20 = R1[12];
            T21 = T19 + T20;
            T43 = T19 - T20;
        }
        T22 = T18 - T21;
        T55 = FMA(KP414213562, T42, T43);
        T34 = T18 + T21;
        T44 = FNMS(KP414213562, T43, T42);
    }
    {
        E T25;
        E T45;
        E T28;
        E T46;
        {
            E T23;
            E T24;
            E T26;
            E T27;
            T23 = R1[14];
            T24 = R1[6];
            T25 = T23 + T24;
            T45 = T23 - T24;
            T26 = R1[2];
            T27 = R1[10];
            T28 = T26 + T27;
            T46 = T27 - T26;
        }
        T29 = T25 - T28;
        T54 = FMA(KP414213562, T45, T46);
        T35 = T25 + T28;
        T47 = FNMS(KP414213562, T46, T45);
    }
    Cr[8] = T7 - T14;
    Ci[8] = T35 - T34;
    {
        E T15;
        E T30;
        E T31;
        E T32;
        T15 = T3 - T6;
        T30 = T22 + T29;
        Cr[12] = FNMS(KP707106781, T30, T15);
        Cr[4] = FMA(KP707106781, T30, T15);
        T31 = T13 - T10;
        T32 = T29 - T22;
        Ci[4] = FMA(KP707106781, T32, T31);
        Ci[12] = FMS(KP707106781, T32, T31);
    }
    {
        E T33;
        E T36;
        E T41;
        E T48;
        T33 = T7 + T14;
        T36 = T34 + T35;
        Cr[16] = T33 - T36;
        Cr[0] = T33 + T36;
        T41 = FMA(KP707106781, T40, T37);
        T48 = T44 + T47;
        Cr[14] = FNMS(KP923879532, T48, T41);
        Cr[2] = FMA(KP923879532, T48, T41);
    }
    T53 = FNMS(KP707106781, T50, T49);
    T56 = T54 - T55;
    Ci[2] = FMS(KP923879532, T56, T53);
    Ci[14] = FMA(KP923879532, T56, T53);
    {
        E T57;
        E T58;
        E T51;
        E T52;
        T57 = FNMS(KP707106781, T40, T37);
        T58 = T55 + T54;
        Cr[10] = FNMS(KP923879532, T58, T57);
        Cr[6] = FMA(KP923879532, T58, T57);
        T51 = FMA(KP707106781, T50, T49);
        T52 = T47 - T44;
        Ci[6] = FMA(KP923879532, T52, T51);
        Ci[10] = FMS(KP923879532, T52, T51);
    }
    Ci[0] = constant(0.0f);
    Ci[16] = constant(0.0f);
}

template <int n>
static void dft(Vec16f data[/* (n - 1) * stride * 2 + 2 */], int stride = 1);

// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 1 -sign -1
template <>
void dft<1>(Vec16f data[/* 2 */], int stride) {
}

// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 3 -sign -1
template <>
void dft<3>(Vec16f data[/* 4 * stride + 2 */], int stride) {
    using E = Vec16f;

    auto ri = &data[0];
    auto ii = &data[1];
    auto ro = ri;
    auto io = ii;

    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);

    E T1;
    E T9;
    E T4;
    E T12;
    E T8;
    E T10;
    E T5;
    E T11;
    T1 = ri[0 * stride];
    T9 = ii[0 * stride];
    {
        E T2;
        E T3;
        E T6;
        E T7;
        T2 = ri[2 * stride];
        T3 = ri[4 * stride];
        T4 = T2 + T3;
        T12 = T3 - T2;
        T6 = ii[2 * stride];
        T7 = ii[4 * stride];
        T8 = T6 - T7;
        T10 = T6 + T7;
    }
    ro[0 * stride] = T1 + T4;
    io[0 * stride] = T9 + T10;
    T5 = FNMS(KP500000000, T4, T1);
    ro[4 * stride] = FNMS(KP866025403, T8, T5);
    ro[2 * stride] = FMA(KP866025403, T8, T5);
    T11 = FNMS(KP500000000, T10, T9);
    io[2 * stride] = FMA(KP866025403, T12, T11);
    io[4 * stride] = FNMS(KP866025403, T12, T11);
}

// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 5 -sign -1
template <>
void dft<5>(Vec16f data[/* 8 * stride + 2 */], int stride) {
    using E = Vec16f;

    auto ri = &data[0];
    auto ii = &data[1];
    auto ro = ri;
    auto io = ii;

    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);

    E T1;
    E T21;
    E T8;
    E T29;
    E T10;
    E T28;
    E T14;
    E T26;
    E T17;
    E T24;
    T1 = ri[0 * stride];
    T21 = ii[0 * stride];
    {
        E T2;
        E T3;
        E T4;
        E T5;
        E T6;
        E T7;
        T2 = ri[2 * stride];
        T3 = ri[8 * stride];
        T4 = T2 + T3;
        T5 = ri[4 * stride];
        T6 = ri[6 * stride];
        T7 = T5 + T6;
        T8 = T4 + T7;
        T29 = T5 - T6;
        T10 = T4 - T7;
        T28 = T2 - T3;
    }
    {
        E T12;
        E T13;
        E T22;
        E T15;
        E T16;
        E T23;
        T12 = ii[2 * stride];
        T13 = ii[8 * stride];
        T22 = T12 + T13;
        T15 = ii[4 * stride];
        T16 = ii[6 * stride];
        T23 = T15 + T16;
        T14 = T12 - T13;
        T26 = T22 - T23;
        T17 = T15 - T16;
        T24 = T22 + T23;
    }
    ro[0 * stride] = T1 + T8;
    io[0 * stride] = T21 + T24;
    {
        E T18;
        E T20;
        E T11;
        E T19;
        E T9;
        T18 = FMA(KP618033988, T17, T14);
        T20 = FNMS(KP618033988, T14, T17);
        T9 = FNMS(KP250000000, T8, T1);
        T11 = FMA(KP559016994, T10, T9);
        T19 = FNMS(KP559016994, T10, T9);
        ro[8 * stride] = FNMS(KP951056516, T18, T11);
        ro[6 * stride] = FMA(KP951056516, T20, T19);
        ro[2 * stride] = FMA(KP951056516, T18, T11);
        ro[4 * stride] = FNMS(KP951056516, T20, T19);
    }
    {
        E T30;
        E T32;
        E T27;
        E T31;
        E T25;
        T30 = FMA(KP618033988, T29, T28);
        T32 = FNMS(KP618033988, T28, T29);
        T25 = FNMS(KP250000000, T24, T21);
        T27 = FMA(KP559016994, T26, T25);
        T31 = FNMS(KP559016994, T26, T25);
        io[2 * stride] = FNMS(KP951056516, T30, T27);
        io[6 * stride] = FNMS(KP951056516, T32, T31);
        io[8 * stride] = FMA(KP951056516, T30, T27);
        io[4 * stride] = FMA(KP951056516, T32, T31);
    }
}

template <>
void dft<7>(Vec16f data[/* 12 * stride + 2 */], int stride) {
    using E = Vec16f;

    auto ri = &data[0];
    auto ii = &data[1];
    auto ro = ri;
    auto io = ii;

    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);

    E T1;
    E T35;
    E T4;
    E T44;
    E T10;
    E T42;
    E T7;
    E T43;
    E T11;
    E T25;
    E T55;
    E T50;
    E T45;
    E T30;
    E T16;
    E T37;
    E T22;
    E T38;
    E T19;
    E T36;
    E T23;
    E T28;
    E T52;
    E T47;
    E T39;
    E T33;
    T1 = ri[0 * stride];
    T35 = ii[0 * stride];
    {
        E T2;
        E T3;
        E T14;
        E T15;
        T2 = ri[2 * stride];
        T3 = ri[12 * stride];
        T4 = T2 + T3;
        T44 = T3 - T2;
        {
            E T8;
            E T9;
            E T5;
            E T6;
            T8 = ri[6 * stride];
            T9 = ri[8 * stride];
            T10 = T8 + T9;
            T42 = T9 - T8;
            T5 = ri[4 * stride];
            T6 = ri[10 * stride];
            T7 = T5 + T6;
            T43 = T6 - T5;
        }
        T11 = FNMS(KP356895867, T7, T4);
        T25 = FNMS(KP356895867, T4, T10);
        T55 = FMA(KP554958132, T42, T44);
        T50 = FMA(KP554958132, T43, T42);
        T45 = FNMS(KP554958132, T44, T43);
        T30 = FNMS(KP356895867, T10, T7);
        T14 = ii[4 * stride];
        T15 = ii[10 * stride];
        T16 = T14 - T15;
        T37 = T14 + T15;
        {
            E T20;
            E T21;
            E T17;
            E T18;
            T20 = ii[6 * stride];
            T21 = ii[8 * stride];
            T22 = T20 - T21;
            T38 = T20 + T21;
            T17 = ii[2 * stride];
            T18 = ii[12 * stride];
            T19 = T17 - T18;
            T36 = T17 + T18;
        }
        T23 = FMA(KP554958132, T22, T19);
        T28 = FMA(KP554958132, T16, T22);
        T52 = FNMS(KP356895867, T37, T36);
        T47 = FNMS(KP356895867, T36, T38);
        T39 = FNMS(KP356895867, T38, T37);
        T33 = FNMS(KP554958132, T19, T16);
    }
    ro[0 * stride] = T1 + T4 + T7 + T10;
    io[0 * stride] = T35 + T36 + T37 + T38;
    {
        E T24;
        E T13;
        E T12;
        E T56;
        E T54;
        E T53;
        T24 = FMA(KP801937735, T23, T16);
        T12 = FNMS(KP692021471, T11, T10);
        T13 = FNMS(KP900968867, T12, T1);
        ro[12 * stride] = FNMS(KP974927912, T24, T13);
        ro[2 * stride] = FMA(KP974927912, T24, T13);
        T56 = FMA(KP801937735, T55, T43);
        T53 = FNMS(KP692021471, T52, T38);
        T54 = FNMS(KP900968867, T53, T35);
        io[2 * stride] = FMA(KP974927912, T56, T54);
        io[12 * stride] = FNMS(KP974927912, T56, T54);
    }
    {
        E T29;
        E T27;
        E T26;
        E T51;
        E T49;
        E T48;
        T29 = FNMS(KP801937735, T28, T19);
        T26 = FNMS(KP692021471, T25, T7);
        T27 = FNMS(KP900968867, T26, T1);
        ro[10 * stride] = FNMS(KP974927912, T29, T27);
        ro[4 * stride] = FMA(KP974927912, T29, T27);
        T51 = FNMS(KP801937735, T50, T44);
        T48 = FNMS(KP692021471, T47, T37);
        T49 = FNMS(KP900968867, T48, T35);
        io[4 * stride] = FMA(KP974927912, T51, T49);
        io[10 * stride] = FNMS(KP974927912, T51, T49);
    }
    {
        E T34;
        E T32;
        E T31;
        E T46;
        E T41;
        E T40;
        T34 = FNMS(KP801937735, T33, T22);
        T31 = FNMS(KP692021471, T30, T4);
        T32 = FNMS(KP900968867, T31, T1);
        ro[8 * stride] = FNMS(KP974927912, T34, T32);
        ro[6 * stride] = FMA(KP974927912, T34, T32);
        T46 = FNMS(KP801937735, T45, T42);
        T40 = FNMS(KP692021471, T39, T36);
        T41 = FNMS(KP900968867, T40, T35);
        io[6 * stride] = FMA(KP974927912, T46, T41);
        io[8 * stride] = FNMS(KP974927912, T46, T41);
    }
}

// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 16 -sign -1
template <>
void dft<16>(Vec16f data[/* 30 * stride + 2 */], int stride) {
    using E = Vec16f;

    auto ri = &data[0];
    auto ii = &data[1];
    auto ro = ri;
    auto io = ii;

    DK(KP923879532, +0.923879532511286756128183189396788286822416626);
    DK(KP414213562, +0.414213562373095048801688724209698078569671875);
    DK(KP707106781, +0.707106781186547524400844362104849039284835938);

    E T7;
    E T115;
    E T129;
    E T38;
    E T49;
    E T95;
    E T105;
    E T83;
    E T29;
    E T126;
    E T141;
    E T73;
    E T78;
    E T102;
    E T123;
    E T101;
    E T14;
    E T116;
    E T130;
    E T45;
    E T52;
    E T84;
    E T85;
    E T55;
    E T22;
    E T121;
    E T140;
    E T62;
    E T67;
    E T99;
    E T118;
    E T98;
    {
        E T3;
        E T47;
        E T34;
        E T82;
        E T6;
        E T81;
        E T37;
        E T48;
        {
            E T1;
            E T2;
            E T32;
            E T33;
            T1 = ri[0 * stride];
            T2 = ri[16 * stride];
            T3 = T1 + T2;
            T47 = T1 - T2;
            T32 = ii[0 * stride];
            T33 = ii[16 * stride];
            T34 = T32 + T33;
            T82 = T32 - T33;
        }
        {
            E T4;
            E T5;
            E T35;
            E T36;
            T4 = ri[8 * stride];
            T5 = ri[24 * stride];
            T6 = T4 + T5;
            T81 = T4 - T5;
            T35 = ii[8 * stride];
            T36 = ii[24 * stride];
            T37 = T35 + T36;
            T48 = T35 - T36;
        }
        T7 = T3 + T6;
        T115 = T3 - T6;
        T129 = T34 - T37;
        T38 = T34 + T37;
        T49 = T47 - T48;
        T95 = T47 + T48;
        T105 = T82 - T81;
        T83 = T81 + T82;
    }
    {
        E T25;
        E T74;
        E T72;
        E T124;
        E T28;
        E T69;
        E T77;
        E T125;
        {
            E T23;
            E T24;
            E T70;
            E T71;
            T23 = ri[30 * stride];
            T24 = ri[14 * stride];
            T25 = T23 + T24;
            T74 = T23 - T24;
            T70 = ii[30 * stride];
            T71 = ii[14 * stride];
            T72 = T70 - T71;
            T124 = T70 + T71;
        }
        {
            E T26;
            E T27;
            E T75;
            E T76;
            T26 = ri[6 * stride];
            T27 = ri[22 * stride];
            T28 = T26 + T27;
            T69 = T26 - T27;
            T75 = ii[6 * stride];
            T76 = ii[22 * stride];
            T77 = T75 - T76;
            T125 = T75 + T76;
        }
        T29 = T25 + T28;
        T126 = T124 - T125;
        T141 = T124 + T125;
        T73 = T69 + T72;
        T78 = T74 - T77;
        T102 = T72 - T69;
        T123 = T25 - T28;
        T101 = T74 + T77;
    }
    {
        E T10;
        E T51;
        E T41;
        E T50;
        E T13;
        E T53;
        E T44;
        E T54;
        {
            E T8;
            E T9;
            E T39;
            E T40;
            T8 = ri[4 * stride];
            T9 = ri[20 * stride];
            T10 = T8 + T9;
            T51 = T8 - T9;
            T39 = ii[4 * stride];
            T40 = ii[20 * stride];
            T41 = T39 + T40;
            T50 = T39 - T40;
        }
        {
            E T11;
            E T12;
            E T42;
            E T43;
            T11 = ri[28 * stride];
            T12 = ri[12 * stride];
            T13 = T11 + T12;
            T53 = T11 - T12;
            T42 = ii[28 * stride];
            T43 = ii[12 * stride];
            T44 = T42 + T43;
            T54 = T42 - T43;
        }
        T14 = T10 + T13;
        T116 = T41 - T44;
        T130 = T13 - T10;
        T45 = T41 + T44;
        T52 = T50 - T51;
        T84 = T53 - T54;
        T85 = T51 + T50;
        T55 = T53 + T54;
    }
    {
        E T18;
        E T63;
        E T61;
        E T119;
        E T21;
        E T58;
        E T66;
        E T120;
        {
            E T16;
            E T17;
            E T59;
            E T60;
            T16 = ri[2 * stride];
            T17 = ri[18 * stride];
            T18 = T16 + T17;
            T63 = T16 - T17;
            T59 = ii[2 * stride];
            T60 = ii[18 * stride];
            T61 = T59 - T60;
            T119 = T59 + T60;
        }
        {
            E T19;
            E T20;
            E T64;
            E T65;
            T19 = ri[10 * stride];
            T20 = ri[26 * stride];
            T21 = T19 + T20;
            T58 = T19 - T20;
            T64 = ii[10 * stride];
            T65 = ii[26 * stride];
            T66 = T64 - T65;
            T120 = T64 + T65;
        }
        T22 = T18 + T21;
        T121 = T119 - T120;
        T140 = T119 + T120;
        T62 = T58 + T61;
        T67 = T63 - T66;
        T99 = T61 - T58;
        T118 = T18 - T21;
        T98 = T63 + T66;
    }
    {
        E T15;
        E T30;
        E T143;
        E T144;
        T15 = T7 + T14;
        T30 = T22 + T29;
        ro[16 * stride] = T15 - T30;
        ro[0 * stride] = T15 + T30;
        T143 = T38 + T45;
        T144 = T140 + T141;
        io[16 * stride] = T143 - T144;
        io[0 * stride] = T143 + T144;
    }
    {
        E T31;
        E T46;
        E T139;
        E T142;
        T31 = T29 - T22;
        T46 = T38 - T45;
        io[8 * stride] = T31 + T46;
        io[24 * stride] = T46 - T31;
        T139 = T7 - T14;
        T142 = T140 - T141;
        ro[24 * stride] = T139 - T142;
        ro[8 * stride] = T139 + T142;
    }
    {
        E T117;
        E T131;
        E T128;
        E T132;
        E T122;
        E T127;
        T117 = T115 + T116;
        T131 = T129 - T130;
        T122 = T118 + T121;
        T127 = T123 - T126;
        T128 = T122 + T127;
        T132 = T127 - T122;
        ro[20 * stride] = FNMS(KP707106781, T128, T117);
        io[12 * stride] = FMA(KP707106781, T132, T131);
        ro[4 * stride] = FMA(KP707106781, T128, T117);
        io[28 * stride] = FNMS(KP707106781, T132, T131);
    }
    {
        E T133;
        E T137;
        E T136;
        E T138;
        E T134;
        E T135;
        T133 = T115 - T116;
        T137 = T130 + T129;
        T134 = T121 - T118;
        T135 = T123 + T126;
        T136 = T134 - T135;
        T138 = T134 + T135;
        ro[28 * stride] = FNMS(KP707106781, T136, T133);
        io[4 * stride] = FMA(KP707106781, T138, T137);
        ro[12 * stride] = FMA(KP707106781, T136, T133);
        io[20 * stride] = FNMS(KP707106781, T138, T137);
    }
    {
        E T57;
        E T93;
        E T87;
        E T89;
        E T80;
        E T88;
        E T92;
        E T94;
        E T56;
        E T86;
        T56 = T52 - T55;
        T57 = FMA(KP707106781, T56, T49);
        T93 = FNMS(KP707106781, T56, T49);
        T86 = T84 - T85;
        T87 = FNMS(KP707106781, T86, T83);
        T89 = FMA(KP707106781, T86, T83);
        {
            E T68;
            E T79;
            E T90;
            E T91;
            T68 = FMA(KP414213562, T67, T62);
            T79 = FNMS(KP414213562, T78, T73);
            T80 = T68 - T79;
            T88 = T68 + T79;
            T90 = FMA(KP414213562, T73, T78);
            T91 = FNMS(KP414213562, T62, T67);
            T92 = T90 - T91;
            T94 = T91 + T90;
        }
        ro[22 * stride] = FNMS(KP923879532, T80, T57);
        io[22 * stride] = FNMS(KP923879532, T92, T89);
        ro[6 * stride] = FMA(KP923879532, T80, T57);
        io[6 * stride] = FMA(KP923879532, T92, T89);
        io[14 * stride] = FNMS(KP923879532, T88, T87);
        ro[14 * stride] = FNMS(KP923879532, T94, T93);
        io[30 * stride] = FMA(KP923879532, T88, T87);
        ro[30 * stride] = FMA(KP923879532, T94, T93);
    }
    {
        E T97;
        E T109;
        E T107;
        E T113;
        E T104;
        E T108;
        E T112;
        E T114;
        E T96;
        E T106;
        T96 = T85 + T84;
        T97 = FMA(KP707106781, T96, T95);
        T109 = FNMS(KP707106781, T96, T95);
        T106 = T52 + T55;
        T107 = FNMS(KP707106781, T106, T105);
        T113 = FMA(KP707106781, T106, T105);
        {
            E T100;
            E T103;
            E T110;
            E T111;
            T100 = FMA(KP414213562, T99, T98);
            T103 = FNMS(KP414213562, T102, T101);
            T104 = T100 + T103;
            T108 = T103 - T100;
            T110 = FNMS(KP414213562, T98, T99);
            T111 = FMA(KP414213562, T101, T102);
            T112 = T110 - T111;
            T114 = T110 + T111;
        }
        ro[18 * stride] = FNMS(KP923879532, T104, T97);
        io[18 * stride] = FNMS(KP923879532, T114, T113);
        ro[2 * stride] = FMA(KP923879532, T104, T97);
        io[2 * stride] = FMA(KP923879532, T114, T113);
        io[26 * stride] = FNMS(KP923879532, T108, T107);
        ro[26 * stride] = FNMS(KP923879532, T112, T109);
        io[10 * stride] = FMA(KP923879532, T108, T107);
        ro[10 * stride] = FMA(KP923879532, T112, T109);
    }
}

template <int n>
static void idft(Vec16f data[/* (n - 1) * stride * 2 + 2 */], int stride = 1);

// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 1 -sign 1
template <>
void idft<1>(Vec16f data[/* 2 */], int stride) {
}

// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 3 -sign 1
template <>
void idft<3>(Vec16f data[/* 4 * stride + 2 */], int stride) {
    using E = Vec16f;

    auto ri = &data[0];
    auto ii = &data[1];
    auto ro = ri;
    auto io = ii;

    DK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);

    E T1;
    E T5;
    E T4;
    E T10;
    E T8;
    E T12;
    E T9;
    E T11;
    T1 = ri[0 * stride];
    T5 = ii[0 * stride];
    {
        E T2;
        E T3;
        E T6;
        E T7;
        T2 = ri[2 * stride];
        T3 = ri[4 * stride];
        T4 = T2 + T3;
        T10 = T2 - T3;
        T6 = ii[2 * stride];
        T7 = ii[4 * stride];
        T8 = T6 + T7;
        T12 = T7 - T6;
    }
    ro[0 * stride] = T1 + T4;
    io[0 * stride] = T5 + T8;
    T9 = FNMS(KP500000000, T8, T5);
    io[2 * stride] = FMA(KP866025403, T10, T9);
    io[4 * stride] = FNMS(KP866025403, T10, T9);
    T11 = FNMS(KP500000000, T4, T1);
    ro[4 * stride] = FNMS(KP866025403, T12, T11);
    ro[2 * stride] = FMA(KP866025403, T12, T11);
}

// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 5 -sign 1
template <>
void idft<5>(Vec16f data[/* 8 * stride + 2 */], int stride) {
    using E = Vec16f;

    auto ri = &data[0];
    auto ii = &data[1];
    auto ro = ri;
    auto io = ii;

    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP618033988, +0.618033988749894848204586834365638117720309180);

    E T1;
    E T21;
    E T8;
    E T29;
    E T10;
    E T28;
    E T14;
    E T26;
    E T17;
    E T24;
    T1 = ri[0 * stride];
    T21 = ii[0 * stride];
    {
        E T2;
        E T3;
        E T4;
        E T5;
        E T6;
        E T7;
        T2 = ri[2 * stride];
        T3 = ri[8 * stride];
        T4 = T2 + T3;
        T5 = ri[4 * stride];
        T6 = ri[6 * stride];
        T7 = T5 + T6;
        T8 = T4 + T7;
        T29 = T5 - T6;
        T10 = T4 - T7;
        T28 = T2 - T3;
    }
    {
        E T12;
        E T13;
        E T22;
        E T15;
        E T16;
        E T23;
        T12 = ii[2 * stride];
        T13 = ii[8 * stride];
        T22 = T12 + T13;
        T15 = ii[4 * stride];
        T16 = ii[6 * stride];
        T23 = T15 + T16;
        T14 = T12 - T13;
        T26 = T22 - T23;
        T17 = T15 - T16;
        T24 = T22 + T23;
    }
    ro[0 * stride] = T1 + T8;
    io[0 * stride] = T21 + T24;
    {
        E T18;
        E T20;
        E T11;
        E T19;
        E T9;
        T18 = FMA(KP618033988, T17, T14);
        T20 = FNMS(KP618033988, T14, T17);
        T9 = FNMS(KP250000000, T8, T1);
        T11 = FMA(KP559016994, T10, T9);
        T19 = FNMS(KP559016994, T10, T9);
        ro[2 * stride] = FNMS(KP951056516, T18, T11);
        ro[4 * stride] = FMA(KP951056516, T20, T19);
        ro[8 * stride] = FMA(KP951056516, T18, T11);
        ro[6 * stride] = FNMS(KP951056516, T20, T19);
    }
    {
        E T30;
        E T32;
        E T27;
        E T31;
        E T25;
        T30 = FMA(KP618033988, T29, T28);
        T32 = FNMS(KP618033988, T28, T29);
        T25 = FNMS(KP250000000, T24, T21);
        T27 = FMA(KP559016994, T26, T25);
        T31 = FNMS(KP559016994, T26, T25);
        io[2 * stride] = FMA(KP951056516, T30, T27);
        io[6 * stride] = FMA(KP951056516, T32, T31);
        io[8 * stride] = FNMS(KP951056516, T30, T27);
        io[4 * stride] = FNMS(KP951056516, T32, T31);
    }
}

// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 7 -sign 1
template <>
void idft<7>(Vec16f data[/* 8 * stride + 2 */], int stride) {
    using E = Vec16f;

    auto ri = &data[0];
    auto ii = &data[1];
    auto ro = ri;
    auto io = ii;

    DK(KP974927912, +0.974927912181823607018131682993931217232785801);
    DK(KP900968867, +0.900968867902419126236102319507445051165919162);
    DK(KP692021471, +0.692021471630095869627814897002069140197260599);
    DK(KP801937735, +0.801937735804838252472204639014890102331838324);
    DK(KP356895867, +0.356895867892209443894399510021300583399127187);
    DK(KP554958132, +0.554958132087371191422194871006410481067288862);

    E T1;
    E T11;
    E T4;
    E T26;
    E T10;
    E T24;
    E T7;
    E T25;
    E T27;
    E T32;
    E T52;
    E T47;
    E T39;
    E T37;
    E T14;
    E T43;
    E T20;
    E T44;
    E T17;
    E T42;
    E T21;
    E T29;
    E T55;
    E T50;
    E T45;
    E T34;
    T1 = ri[0 * stride];
    T11 = ii[0 * stride];
    {
        E T2;
        E T3;
        E T12;
        E T13;
        T2 = ri[2 * stride];
        T3 = ri[12 * stride];
        T4 = T2 + T3;
        T26 = T2 - T3;
        {
            E T8;
            E T9;
            E T5;
            E T6;
            T8 = ri[6 * stride];
            T9 = ri[8 * stride];
            T10 = T8 + T9;
            T24 = T8 - T9;
            T5 = ri[4 * stride];
            T6 = ri[10 * stride];
            T7 = T5 + T6;
            T25 = T5 - T6;
        }
        T27 = FNMS(KP554958132, T26, T25);
        T32 = FMA(KP554958132, T25, T24);
        T52 = FNMS(KP356895867, T10, T7);
        T47 = FNMS(KP356895867, T4, T10);
        T39 = FNMS(KP356895867, T7, T4);
        T37 = FMA(KP554958132, T24, T26);
        T12 = ii[2 * stride];
        T13 = ii[12 * stride];
        T14 = T12 + T13;
        T43 = T13 - T12;
        {
            E T18;
            E T19;
            E T15;
            E T16;
            T18 = ii[6 * stride];
            T19 = ii[8 * stride];
            T20 = T18 + T19;
            T44 = T19 - T18;
            T15 = ii[4 * stride];
            T16 = ii[10 * stride];
            T17 = T15 + T16;
            T42 = T16 - T15;
        }
        T21 = FNMS(KP356895867, T20, T17);
        T29 = FNMS(KP356895867, T14, T20);
        T55 = FNMS(KP554958132, T43, T42);
        T50 = FMA(KP554958132, T42, T44);
        T45 = FMA(KP554958132, T44, T43);
        T34 = FNMS(KP356895867, T17, T14);
    }
    ro[0 * stride] = T1 + T4 + T7 + T10;
    io[0 * stride] = T11 + T14 + T17 + T20;
    {
        E T28;
        E T23;
        E T22;
        E T56;
        E T54;
        E T53;
        T28 = FNMS(KP801937735, T27, T24);
        T22 = FNMS(KP692021471, T21, T14);
        T23 = FNMS(KP900968867, T22, T11);
        io[6 * stride] = FMA(KP974927912, T28, T23);
        io[8 * stride] = FNMS(KP974927912, T28, T23);
        T56 = FNMS(KP801937735, T55, T44);
        T53 = FNMS(KP692021471, T52, T4);
        T54 = FNMS(KP900968867, T53, T1);
        ro[8 * stride] = FNMS(KP974927912, T56, T54);
        ro[6 * stride] = FMA(KP974927912, T56, T54);
    }
    {
        E T33;
        E T31;
        E T30;
        E T51;
        E T49;
        E T48;
        T33 = FNMS(KP801937735, T32, T26);
        T30 = FNMS(KP692021471, T29, T17);
        T31 = FNMS(KP900968867, T30, T11);
        io[4 * stride] = FMA(KP974927912, T33, T31);
        io[10 * stride] = FNMS(KP974927912, T33, T31);
        T51 = FNMS(KP801937735, T50, T43);
        T48 = FNMS(KP692021471, T47, T7);
        T49 = FNMS(KP900968867, T48, T1);
        ro[10 * stride] = FNMS(KP974927912, T51, T49);
        ro[4 * stride] = FMA(KP974927912, T51, T49);
    }
    {
        E T38;
        E T36;
        E T35;
        E T46;
        E T41;
        E T40;
        T38 = FMA(KP801937735, T37, T25);
        T35 = FNMS(KP692021471, T34, T20);
        T36 = FNMS(KP900968867, T35, T11);
        io[2 * stride] = FMA(KP974927912, T38, T36);
        io[12 * stride] = FNMS(KP974927912, T38, T36);
        T46 = FMA(KP801937735, T45, T42);
        T40 = FNMS(KP692021471, T39, T10);
        T41 = FNMS(KP900968867, T40, T1);
        ro[12 * stride] = FNMS(KP974927912, T46, T41);
        ro[2 * stride] = FMA(KP974927912, T46, T41);
    }
}

// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 16 -sign 1
template <>
void idft<16>(Vec16f data[/* 30 * stride + 2 */], int stride) {
    using E = Vec16f;

    auto ri = &data[0];
    auto ii = &data[1];
    auto ro = ri;
    auto io = ii;

    DK(KP923879532, +0.923879532511286756128183189396788286822416626);
    DK(KP414213562, +0.414213562373095048801688724209698078569671875);
    DK(KP707106781, +0.707106781186547524400844362104849039284835938);

    E T7;
    E T115;
    E T129;
    E T38;
    E T49;
    E T95;
    E T105;
    E T83;
    E T29;
    E T126;
    E T141;
    E T73;
    E T78;
    E T99;
    E T123;
    E T98;
    E T14;
    E T116;
    E T130;
    E T45;
    E T52;
    E T84;
    E T85;
    E T55;
    E T22;
    E T121;
    E T140;
    E T62;
    E T67;
    E T102;
    E T118;
    E T101;
    {
        E T3;
        E T81;
        E T34;
        E T48;
        E T6;
        E T47;
        E T37;
        E T82;
        {
            E T1;
            E T2;
            E T32;
            E T33;
            T1 = ri[0 * stride];
            T2 = ri[16 * stride];
            T3 = T1 + T2;
            T81 = T1 - T2;
            T32 = ii[0 * stride];
            T33 = ii[16 * stride];
            T34 = T32 + T33;
            T48 = T32 - T33;
        }
        {
            E T4;
            E T5;
            E T35;
            E T36;
            T4 = ri[8 * stride];
            T5 = ri[24 * stride];
            T6 = T4 + T5;
            T47 = T4 - T5;
            T35 = ii[8 * stride];
            T36 = ii[24 * stride];
            T37 = T35 + T36;
            T82 = T35 - T36;
        }
        T7 = T3 + T6;
        T115 = T34 - T37;
        T129 = T3 - T6;
        T38 = T34 + T37;
        T49 = T47 + T48;
        T95 = T81 + T82;
        T105 = T48 - T47;
        T83 = T81 - T82;
    }
    {
        E T25;
        E T74;
        E T72;
        E T124;
        E T28;
        E T69;
        E T77;
        E T125;
        {
            E T23;
            E T24;
            E T70;
            E T71;
            T23 = ri[30 * stride];
            T24 = ri[14 * stride];
            T25 = T23 + T24;
            T74 = T23 - T24;
            T70 = ii[30 * stride];
            T71 = ii[14 * stride];
            T72 = T70 - T71;
            T124 = T70 + T71;
        }
        {
            E T26;
            E T27;
            E T75;
            E T76;
            T26 = ri[6 * stride];
            T27 = ri[22 * stride];
            T28 = T26 + T27;
            T69 = T26 - T27;
            T75 = ii[6 * stride];
            T76 = ii[22 * stride];
            T77 = T75 - T76;
            T125 = T75 + T76;
        }
        T29 = T25 + T28;
        T126 = T124 - T125;
        T141 = T124 + T125;
        T73 = T69 + T72;
        T78 = T74 - T77;
        T99 = T74 + T77;
        T123 = T25 - T28;
        T98 = T72 - T69;
    }
    {
        E T10;
        E T50;
        E T41;
        E T51;
        E T13;
        E T54;
        E T44;
        E T53;
        {
            E T8;
            E T9;
            E T39;
            E T40;
            T8 = ri[4 * stride];
            T9 = ri[20 * stride];
            T10 = T8 + T9;
            T50 = T8 - T9;
            T39 = ii[4 * stride];
            T40 = ii[20 * stride];
            T41 = T39 + T40;
            T51 = T39 - T40;
        }
        {
            E T11;
            E T12;
            E T42;
            E T43;
            T11 = ri[28 * stride];
            T12 = ri[12 * stride];
            T13 = T11 + T12;
            T54 = T11 - T12;
            T42 = ii[28 * stride];
            T43 = ii[12 * stride];
            T44 = T42 + T43;
            T53 = T42 - T43;
        }
        T14 = T10 + T13;
        T116 = T10 - T13;
        T130 = T44 - T41;
        T45 = T41 + T44;
        T52 = T50 + T51;
        T84 = T50 - T51;
        T85 = T54 + T53;
        T55 = T53 - T54;
    }
    {
        E T18;
        E T63;
        E T61;
        E T119;
        E T21;
        E T58;
        E T66;
        E T120;
        {
            E T16;
            E T17;
            E T59;
            E T60;
            T16 = ri[2 * stride];
            T17 = ri[18 * stride];
            T18 = T16 + T17;
            T63 = T16 - T17;
            T59 = ii[2 * stride];
            T60 = ii[18 * stride];
            T61 = T59 - T60;
            T119 = T59 + T60;
        }
        {
            E T19;
            E T20;
            E T64;
            E T65;
            T19 = ri[10 * stride];
            T20 = ri[26 * stride];
            T21 = T19 + T20;
            T58 = T19 - T20;
            T64 = ii[10 * stride];
            T65 = ii[26 * stride];
            T66 = T64 - T65;
            T120 = T64 + T65;
        }
        T22 = T18 + T21;
        T121 = T119 - T120;
        T140 = T119 + T120;
        T62 = T58 + T61;
        T67 = T63 - T66;
        T102 = T63 + T66;
        T118 = T18 - T21;
        T101 = T61 - T58;
    }
    {
        E T15;
        E T30;
        E T139;
        E T142;
        T15 = T7 + T14;
        T30 = T22 + T29;
        ro[16 * stride] = T15 - T30;
        ro[0 * stride] = T15 + T30;
        T139 = T38 + T45;
        T142 = T140 + T141;
        io[16 * stride] = T139 - T142;
        io[0 * stride] = T139 + T142;
    }
    {
        E T31;
        E T46;
        E T143;
        E T144;
        T31 = T22 - T29;
        T46 = T38 - T45;
        io[8 * stride] = T31 + T46;
        io[24 * stride] = T46 - T31;
        T143 = T7 - T14;
        T144 = T141 - T140;
        ro[24 * stride] = T143 - T144;
        ro[8 * stride] = T143 + T144;
    }
    {
        E T117;
        E T131;
        E T128;
        E T132;
        E T122;
        E T127;
        T117 = T115 - T116;
        T131 = T129 + T130;
        T122 = T118 - T121;
        T127 = T123 + T126;
        T128 = T122 - T127;
        T132 = T122 + T127;
        io[28 * stride] = FNMS(KP707106781, T128, T117);
        ro[4 * stride] = FMA(KP707106781, T132, T131);
        io[12 * stride] = FMA(KP707106781, T128, T117);
        ro[20 * stride] = FNMS(KP707106781, T132, T131);
    }
    {
        E T133;
        E T137;
        E T136;
        E T138;
        E T134;
        E T135;
        T133 = T116 + T115;
        T137 = T129 - T130;
        T134 = T118 + T121;
        T135 = T126 - T123;
        T136 = T134 + T135;
        T138 = T135 - T134;
        io[20 * stride] = FNMS(KP707106781, T136, T133);
        ro[12 * stride] = FMA(KP707106781, T138, T137);
        io[4 * stride] = FMA(KP707106781, T136, T133);
        ro[28 * stride] = FNMS(KP707106781, T138, T137);
    }
    {
        E T57;
        E T89;
        E T87;
        E T93;
        E T80;
        E T88;
        E T92;
        E T94;
        E T56;
        E T86;
        T56 = T52 + T55;
        T57 = FMA(KP707106781, T56, T49);
        T89 = FNMS(KP707106781, T56, T49);
        T86 = T84 + T85;
        T87 = FNMS(KP707106781, T86, T83);
        T93 = FMA(KP707106781, T86, T83);
        {
            E T68;
            E T79;
            E T90;
            E T91;
            T68 = FMA(KP414213562, T67, T62);
            T79 = FNMS(KP414213562, T78, T73);
            T80 = T68 + T79;
            T88 = T79 - T68;
            T90 = FNMS(KP414213562, T62, T67);
            T91 = FMA(KP414213562, T73, T78);
            T92 = T90 - T91;
            T94 = T90 + T91;
        }
        io[18 * stride] = FNMS(KP923879532, T80, T57);
        ro[18 * stride] = FNMS(KP923879532, T94, T93);
        io[2 * stride] = FMA(KP923879532, T80, T57);
        ro[2 * stride] = FMA(KP923879532, T94, T93);
        ro[26 * stride] = FNMS(KP923879532, T88, T87);
        io[26 * stride] = FNMS(KP923879532, T92, T89);
        ro[10 * stride] = FMA(KP923879532, T88, T87);
        io[10 * stride] = FMA(KP923879532, T92, T89);
    }
    {
        E T97;
        E T113;
        E T107;
        E T109;
        E T104;
        E T108;
        E T112;
        E T114;
        E T96;
        E T106;
        T96 = T55 - T52;
        T97 = FMA(KP707106781, T96, T95);
        T113 = FNMS(KP707106781, T96, T95);
        T106 = T84 - T85;
        T107 = FNMS(KP707106781, T106, T105);
        T109 = FMA(KP707106781, T106, T105);
        {
            E T100;
            E T103;
            E T110;
            E T111;
            T100 = FMA(KP414213562, T99, T98);
            T103 = FNMS(KP414213562, T102, T101);
            T104 = T100 - T103;
            T108 = T103 + T100;
            T110 = FMA(KP414213562, T101, T102);
            T111 = FNMS(KP414213562, T98, T99);
            T112 = T110 - T111;
            T114 = T110 + T111;
        }
        ro[22 * stride] = FNMS(KP923879532, T104, T97);
        io[22 * stride] = FNMS(KP923879532, T112, T109);
        ro[6 * stride] = FMA(KP923879532, T104, T97);
        io[6 * stride] = FMA(KP923879532, T112, T109);
        io[14 * stride] = FNMS(KP923879532, T108, T107);
        ro[14 * stride] = FNMS(KP923879532, T114, T113);
        io[30 * stride] = FMA(KP923879532, T108, T107);
        ro[30 * stride] = FMA(KP923879532, T114, T113);
    }
}

template <int n>
static void irdft(Vec16f data[(n / 2 + 1) * 2]);

// ./gen_r2cb.native -standalone -with-rs 2 -with-csr 2 -with-csi 2 -fma -n 16
template <>
void irdft<16>(Vec16f data[18]) {
    using E = Vec16f;

    auto R0 = &data[0];
    auto R1 = &data[1];
    auto Cr = R0;
    auto Ci = R1;

    DK(KP1_847759065, +1.847759065022573512256366378793576573644833252);
    DK(KP414213562, +0.414213562373095048801688724209698078569671875);
    DK(KP1_414213562, +1.414213562373095048801688724209698078569671875);
    DK(KP2_000000000, +2.000000000000000000000000000000000000000000000);

    E T5;
    E T47;
    E T19;
    E T39;
    E T8;
    E T48;
    E T24;
    E T40;
    E T12;
    E T51;
    E T15;
    E T52;
    E T30;
    E T35;
    E T53;
    E T50;
    E T43;
    E T42;
    {
        E T4;
        E T18;
        E T3;
        E T17;
        E T1;
        E T2;
        T4 = Cr[8];
        T18 = Ci[8];
        T1 = Cr[0];
        T2 = Cr[16];
        T3 = T1 + T2;
        T17 = T1 - T2;
        T5 = FMA(KP2_000000000, T4, T3);
        T47 = FNMS(KP2_000000000, T4, T3);
        T19 = FMA(KP2_000000000, T18, T17);
        T39 = FNMS(KP2_000000000, T18, T17);
    }
    {
        E T6;
        E T7;
        E T20;
        E T21;
        E T22;
        E T23;
        T6 = Cr[4];
        T7 = Cr[12];
        T20 = T6 - T7;
        T21 = Ci[4];
        T22 = Ci[12];
        T23 = T21 + T22;
        T8 = T6 + T7;
        T48 = T21 - T22;
        T24 = T20 + T23;
        T40 = T23 - T20;
    }
    {
        E T26;
        E T33;
        E T34;
        E T29;
        {
            E T10;
            E T11;
            E T31;
            E T32;
            T10 = Cr[2];
            T11 = Cr[14];
            T12 = T10 + T11;
            T26 = T10 - T11;
            T31 = Ci[2];
            T32 = Ci[14];
            T33 = T31 + T32;
            T51 = T31 - T32;
        }
        {
            E T13;
            E T14;
            E T27;
            E T28;
            T13 = Cr[10];
            T14 = Cr[6];
            T15 = T13 + T14;
            T34 = T13 - T14;
            T27 = Ci[10];
            T28 = Ci[6];
            T29 = T27 + T28;
            T52 = T27 - T28;
        }
        T30 = T26 + T29;
        T35 = T33 - T34;
        T53 = T51 - T52;
        T50 = T12 - T15;
        T43 = T34 + T33;
        T42 = T26 - T29;
    }
    {
        E T9;
        E T16;
        E T55;
        E T56;
        T9 = FMA(KP2_000000000, T8, T5);
        T16 = T12 + T15;
        R0[8] = FNMS(KP2_000000000, T16, T9);
        R0[0] = FMA(KP2_000000000, T16, T9);
        T55 = FNMS(KP2_000000000, T48, T47);
        T56 = T53 - T50;
        R0[14] = FNMS(KP1_414213562, T56, T55);
        R0[6] = FMA(KP1_414213562, T56, T55);
    }
    {
        E T57;
        E T58;
        E T25;
        E T36;
        T57 = FNMS(KP2_000000000, T8, T5);
        T58 = T52 + T51;
        R0[12] = FNMS(KP2_000000000, T58, T57);
        R0[4] = FMA(KP2_000000000, T58, T57);
        T25 = FMA(KP1_414213562, T24, T19);
        T36 = FMA(KP414213562, T35, T30);
        R1[8] = FNMS(KP1_847759065, T36, T25);
        R1[0] = FMA(KP1_847759065, T36, T25);
    }
    {
        E T37;
        E T38;
        E T45;
        E T46;
        T37 = FNMS(KP1_414213562, T24, T19);
        T38 = FNMS(KP414213562, T30, T35);
        R1[12] = FNMS(KP1_847759065, T38, T37);
        R1[4] = FMA(KP1_847759065, T38, T37);
        T45 = FMA(KP1_414213562, T40, T39);
        T46 = FMA(KP414213562, T42, T43);
        R1[10] = FNMS(KP1_847759065, T46, T45);
        R1[2] = FMA(KP1_847759065, T46, T45);
    }
    {
        E T49;
        E T54;
        E T41;
        E T44;
        T49 = FMA(KP2_000000000, T48, T47);
        T54 = T50 + T53;
        R0[10] = FNMS(KP1_414213562, T54, T49);
        R0[2] = FMA(KP1_414213562, T54, T49);
        T41 = FNMS(KP1_414213562, T40, T39);
        T44 = FNMS(KP414213562, T43, T42);
        R1[6] = FNMS(KP1_847759065, T44, T41);
        R1[14] = FMA(KP1_847759065, T44, T41);
    }
}

template <int n>
static void post_irdft(Vec16f data[n]);

template <>
void post_irdft<16>(Vec16f data[16]) {
    #pragma GCC unroll 7
    for (int i = 1; i < 8; i++) {
        auto temp = data[i];
        data[i] = data[16 - i];
        data[16 - i] = temp;
    }
}

template <int stride = 1>
static inline void transpose_16x16(Vec16f block[/* 16 */]) {
    #pragma GCC unroll 2
    for (int i = 0; i < 2; i++) {
        #pragma GCC unroll 2
        for (int j = 0; j < 2; j++) {
            #pragma GCC unroll 2
            for (int k = 0; k < 2; k++) {
                auto id1 = ((i * 2 + j) * 2 + k) * 2 * stride;
                auto id2 = (((i * 2 + j) * 2 + k) * 2 + 1) * stride;

#if __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)
                Vec16f temp1 = __builtin_shufflevector(block[id1], block[id2], 0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30);
                Vec16f temp2 = __builtin_shufflevector(block[id1], block[id2], 1, 3, 17, 19, 5, 7, 21, 23, 9, 11, 25, 27, 13, 15, 29, 31);
#else // __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)
                Vec16i mask1 = { 0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30 };
                Vec16f temp1 = __builtin_shuffle(block[id1], block[id2], mask1);
                Vec16i mask2 = { 1, 3, 17, 19, 5, 7, 21, 23, 9, 11, 25, 27, 13, 15, 29, 31 };
                Vec16f temp2 = __builtin_shuffle(block[id1], block[id2], mask2);
#endif // __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)

                block[id1] = temp1;
                block[id2] = temp2;
            }

            #pragma GCC unroll 2
            for (int k = 0; k < 2; k++) {
                auto id1 = (((i * 2 + j) * 2) * 2 + k) * stride;
                auto id2 = (((i * 2 + j) * 2 + 1) * 2 + k) * stride;

#if __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)
                Vec16f temp1 = __builtin_shufflevector(block[id1], block[id2], 0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30);
                Vec16f temp2 = __builtin_shufflevector(block[id1], block[id2], 1, 3, 17, 19, 5, 7, 21, 23, 9, 11, 25, 27, 13, 15, 29, 31);
#else // __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)
                Vec16i mask1 = { 0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30 };
                Vec16f temp1 = __builtin_shuffle(block[id1], block[id2], mask1);
                Vec16i mask2 = { 1, 3, 17, 19, 5, 7, 21, 23, 9, 11, 25, 27, 13, 15, 29, 31 };
                Vec16f temp2 = __builtin_shuffle(block[id1], block[id2], mask2);
#endif // __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)

                block[id1] = temp1;
                block[id2] = temp2;
            }
        }

        #pragma GCC unroll 4
        for (int j = 0; j < 4; j++) {
            auto id1 = (i * 8 + j) * stride;
            auto id2 = (i * 8 + 4 + j) * stride;

#if __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)
                Vec16f temp1 = __builtin_shufflevector(block[id1], block[id2], 0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
                Vec16f temp2 = __builtin_shufflevector(block[id1], block[id2], 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);
#else // __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)
                Vec16i mask1 = { 0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27 };
                Vec16f temp1 = __builtin_shuffle(block[id1], block[id2], mask1);
                Vec16i mask2 = { 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31 };
                Vec16f temp2 = __builtin_shuffle(block[id1], block[id2], mask2);
#endif // __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)

            block[id1] = temp1;
            block[id2] = temp2;
        }
    }

    #pragma GCC unroll 8
    for (int i = 0; i < 8; i++) {
#if __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)
                Vec16f temp1 = __builtin_shufflevector(block[i * stride], block[(i + 8) * stride], 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                Vec16f temp2 = __builtin_shufflevector(block[i * stride], block[(i + 8) * stride], 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
#else // __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)
                Vec16i mask1 = { 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23 };
                Vec16f temp1 = __builtin_shuffle(block[i * stride], block[(i + 8) * stride], mask1);
                Vec16i mask2 = { 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 };
                Vec16f temp2 = __builtin_shuffle(block[i * stride], block[(i + 8) * stride], mask2);
#endif // __clang__ || (defined __GNUC_MAJOR__ && __GNUC_MAJOR__ >= 12)

        block[i * stride] = temp1;
        block[(i + 8) * stride] = temp2;
    }
}

template <int stride = 1>
static inline void transpose_32x16(Vec16f block[/* 16 * 2 */]) {
    transpose_16x16<2 * stride>(block);
    transpose_16x16<2 * stride>(block + stride);
}

static inline void remove_mean(
    Vec16f * __restrict block,
    float gf,
    const Vec16f * __restrict window_freq,
    int radius
) {

    for (int i = 0; i < (2 * radius + 1) * 32; i++) {
        block[i] -= gf * window_freq[i];
    }
}

static inline void frequency_filtering(
    Vec16f * __restrict block,
    const Vec16f * __restrict sigma,
    float sigma2,
    float pmin,
    float pmax,
    int filter_type,
    int radius
) {

    if (filter_type == 0) {
        for (int i = 0; i < (2 * radius + 1) * 16; i++) {
            auto psd = square(block[i * 2]) + square(block[i * 2 + 1]);
            auto multiplier = max((psd - sigma[i]) / (psd + 1e-15f), constant(0.0f));
            block[i * 2] *= multiplier;
            block[i * 2 + 1] *= multiplier;
        }
    }
    if (filter_type == 1) {
        for (int i = 0; i < (2 * radius + 1) * 16; i++) {
            auto psd = square(block[i * 2]) + square(block[i * 2 + 1]);
            auto mask = psd < sigma[i];
            block[i * 2] = mask ? 0.0f : block[i * 2];
            block[i * 2 + 1] = mask ? 0.0f : block[i * 2 + 1];
        }
    }
    if (filter_type == 2) {
        for (int i = 0; i < (2 * radius + 1) * 16; i++) {
            block[i * 2] *= sigma[i];
            block[i * 2 + 1] *= sigma[i];
        }
    }
    if (filter_type == 3) {
        for (int i = 0; i < (2 * radius + 1) * 16; i++) {
            auto psd = square(block[i * 2]) + square(block[i * 2 + 1]);
            auto multiplier = (psd >= pmin && psd <= pmax) ? sigma[i] : sigma2;
            block[i * 2] *= multiplier;
            block[i * 2 + 1] *= multiplier;
        }
    }
    if (filter_type == 4) {
        for (int i = 0; i < (2 * radius + 1) * 16; i++) {
            auto psd = square(block[i * 2]) + square(block[i * 2 + 1]);
            auto multiplier = sigma[i] * sqrt(psd * (pmax / ((psd + pmin) * (psd + pmax) + 1e-15f)));
            block[i * 2] *= multiplier;
            block[i * 2 + 1] *= multiplier;
        }
    }
    if (filter_type == 5) {
        for (int i = 0; i < (2 * radius + 1) * 16; i++) {
            auto psd = square(block[i * 2]) + square(block[i * 2 + 1]);
            auto multiplier = pow(max((psd - sigma[i]) / (psd + 1e-15f), constant(0.0f)), constant(pmin));
            block[i * 2] *= multiplier;
            block[i * 2 + 1] *= multiplier;
        }
    }
    if (filter_type == 6) {
        for (int i = 0; i < (2 * radius + 1) * 16; i++) {
            auto psd = square(block[i * 2]) + square(block[i * 2 + 1]);
            auto multiplier = sqrt(max((psd - sigma[i]) / (psd + 1e-15f), constant(0.0f)));
            block[i * 2] *= multiplier;
            block[i * 2 + 1] *= multiplier;
        }
    }
}

static inline void add_mean(
    Vec16f * __restrict block,
    float gf,
    const Vec16f * __restrict window_freq,
    int radius
) {

    for (int i = 0; i < (2 * radius + 1) * 32; i++) {
        block[i] += gf * window_freq[i];
    }
}

static inline void fused(
    Vec16f * __restrict block,
    const Vec16f * __restrict sigma,
    float sigma2,
    float pmin,
    float pmax,
    int filter_type,
    bool zero_mean,
    const Vec16f * __restrict window_freq,
    int radius
) {

    for (int i = 0; i < 2 * radius + 1; i++) {
        transpose_16x16(&block[i * 32]);
        rdft<16>(&block[i * 32]);
        transpose_32x16(&block[i * 32]);
        dft<16>(&block[i * 32]);
    }
    if (radius == 0) {
        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            dft<1>(&block[i * 2], 16);
        }
    }
    if (radius == 1) {
        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            dft<3>(&block[i * 2], 16);
        }
    }
    if (radius == 2) {
        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            dft<5>(&block[i * 2], 16);
        }
    }
    if (radius == 3) {
        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            dft<7>(&block[i * 2], 16);
        }
    }

    float gf {};
    if (zero_mean) {
        gf = block[0][0] / window_freq[0][0];
        remove_mean(block, gf, window_freq, radius);
    }

    frequency_filtering(block, sigma, sigma2, pmin, pmax, filter_type, radius);

    if (zero_mean) {
        add_mean(block, gf, window_freq, radius);
    }

    if (radius == 0) {
        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            idft<1>(&block[i * 2], 16);
        }
    }
    if (radius == 1) {
        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            idft<3>(&block[i * 2], 16);
        }
    }
    if (radius == 2) {
        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            idft<5>(&block[i * 2], 16);
        }
    }
    if (radius == 3) {
        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            idft<7>(&block[i * 2], 16);
        }
    }
    idft<16>(&block[radius * 32]);
    transpose_32x16(&block[radius * 32]);
    irdft<16>(&block[radius * 32]);
    post_irdft<16>(&block[radius * 32]);
    transpose_16x16(&block[radius * 32]);
}

#endif // KERNEL_HPP
