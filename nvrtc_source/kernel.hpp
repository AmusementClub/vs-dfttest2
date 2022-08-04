#ifndef KERNEL_HPP
#define KERNEL_HPP

static const auto fft_header = R"""(
#define K(x) ((E) x)
#define DK(name, value) const E name = K(value)
#define FMA(a, b, c) (((a) * (b)) + (c))
#define FMS(a, b, c) (((a) * (b)) - (c))
#define FNMA(a, b, c) (- (((a) * (b)) + (c)))
#define FNMS(a, b, c) ((c) - ((a) * (b)))

template <int n>
__device__
static void rdft(float data[(n / 2 + 1) * 2]);

template <int n>
__device__
static void dft(float data[/* (n - 1) * stride * 2 + 2 */], int stride = 1);

template <int n>
__device__
static void idft(float data[/* (n - 1) * stride * 2 + 2 */], int stride = 1);

template <int n>
__device__
static void irdft(float data[(n / 2 + 1) * 2]);
)""";

static const char * rdft_implementations[] { R"""(
// ./gen_r2cf.native -standalone -with-rs 2 -with-csr 2 -with-csi 2 -fma -n 16
template <>
__device__
[[maybe_unused]]
void rdft<16>(float data[18]) {
    using E = float;

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
    Ci[0] = 0.0f;
    Ci[16] = 0.0f;
}

)""" };

static const char * dft_implementations[] { R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 1 -sign -1
template <>
__device__
[[maybe_unused]]
void dft<1>(float data[2], int stride) {
}

)""", R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 3 -sign -1
template <>
__device__
[[maybe_unused]]
void dft<3>(float data[/* 4 * stride + 2 */], int stride) {
    using E = float;

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

)""", R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 5 -sign -1
template <>
__device__
[[maybe_unused]]
void dft<5>(float data[/* 8 * stride + 2 */], int stride) {
    using E = float;

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

)""", R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 7 -sign -1
template <>
__device__
[[maybe_unused]]
void dft<7>(float data[/* 12 * stride + 2 */], int stride) {
    using E = float;

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

)""", R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 16 -sign -1
template <>
__device__
[[maybe_unused]]
void dft<16>(float data[/* 30 * stride + 2 */], int stride) {
    using E = float;

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
)""" };

static const char * idft_implementations[] { R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 1 -sign 1
template <>
__device__
[[maybe_unused]]
void idft<1>(float data[2], int stride) {
}

)""", R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 3 -sign 1
template <>
__device__
[[maybe_unused]]
void idft<3>(float data[/* 4 * stride + 2 */], int stride) {
    using E = float;

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

)""", R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 5 -sign 1
template <>
__device__
[[maybe_unused]]
void idft<5>(float data[/* 8 * stride + 2 */], int stride) {
    using E = float;

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

)""", R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 7 -sign 1
template <>
__device__
[[maybe_unused]]
void idft<7>(float data[/* 8 * stride + 2 */], int stride) {
    using E = float;

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

)""", R"""(
// ./gen_notw.native -standalone -with-istride 2 -with-ostride 2 -fma -n 16 -sign 1
template <>
__device__
[[maybe_unused]]
void idft<16>(float data[/* 30 * stride + 2 */], int stride) {
    using E = float;

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
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        data[i * stride] = 0.0f;
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

)""" };

static const char * irdft_implementations[] { R"""(
// ./gen_r2cb.native -standalone -with-rs 2 -with-csr 2 -with-csi 2 -fma -n 16
template <>
__device__
[[maybe_unused]]
void irdft<16>(float data[18]) {
    using E = float;

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
)""" };

static const auto kernel_implementation = R"""(
__device__
extern void filter(float2 & value, int x, int y, int z);

// ZERO_MEAN
// RADIUS
// BLOCK_SIZE
// BLOCK_STEP
// WARPS_PER_BLOCK
// WARP_SIZE
// TYPE
// SCALE
// PEAK (optional)

#if ZERO_MEAN
// __device__ const float window_freq[]; // frequency response of the window
#endif // ZERO_MEAN

__device__
static int calc_pad_size(int size, int block_size, int block_step) {
    return size + ((size % block_size) ? block_size - size % block_size : 0) + max(block_size - block_step, block_step) * 2;
}

__device__
static int calc_pad_num(int size, int block_size, int block_step) {
    return (calc_pad_size(size, block_size, block_step) - block_size) / block_step + 1;
}

__device__
static float to_float(TYPE x) {
    return static_cast<float>(x) * static_cast<float>(SCALE);
}

__device__
static TYPE from_float(float x) {
#ifdef PEAK
    x /= static_cast<float>(SCALE);
    x = fmaxf(0.0f, fminf(x, static_cast<float>(PEAK)));
    return static_cast<TYPE>(__float2int_rz(x + 0.5f));
#else // PEAK // only integral types define it
    return static_cast<TYPE>(x / static_cast<float>(SCALE));
#endif // PEAK
}

// im2col + rdft + frequency_filtering + irdft
extern "C"
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
__global__
void fused(
    float * __restrict__ dstp, // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    const TYPE * __restrict__ srcp, // shape: (2*radius+1, vertical_size, horizontal_size)
    int width,
    int height
) {

    constexpr int radius = static_cast<int>(RADIUS);
    constexpr int block_size = static_cast<int>(BLOCK_SIZE);
    constexpr int block_step = static_cast<int>(BLOCK_STEP);

    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int num_blocks = vertical_num * horizontal_num;

    constexpr int warp_size = static_cast<int>(WARP_SIZE);
    constexpr int warps_per_block = static_cast<int>(WARPS_PER_BLOCK);
    constexpr int transpose_stride = (warp_size % block_size == 0) ? block_size + 1 : block_size;
    __shared__ float2 shared_transpose_buffer[warps_per_block * block_size * transpose_stride];

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    auto transpose_buffer = &shared_transpose_buffer[warp_id * block_size * transpose_stride];

    for (int block_id = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE; block_id < num_blocks; block_id += gridDim.x * WARPS_PER_BLOCK) {
        int ix = block_id % horizontal_num;
        int iy = block_id / horizontal_num;

        if (lane_id < block_size) {
            constexpr int active_mask = (1 << block_size) - 1;
            float2 thread_data[(2 * radius + 1) * block_size];

            // im2col
            #pragma unroll
            for (int i = 0; i < 2 * radius + 1; i++) {
                auto src = &srcp[(i * vertical_size + iy * block_step) * horizontal_size + ix * block_step];
                auto local_thread_data = &thread_data[i * block_size];
                #pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float *) local_thread_data)[j] = to_float(src[j * horizontal_size + lane_id]) * window[(i * block_size + j) * block_size + lane_id];
                }
            }

            // rdft
            #pragma unroll
            for (int i = 0; i < 2 * radius + 1; i++) {
                auto local_thread_data = &thread_data[i * block_size];

                __syncwarp(active_mask);
                // transpose store of real data
                #pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float *) transpose_buffer)[j * transpose_stride + lane_id] = ((float *) local_thread_data)[j];
                }

                __syncwarp(active_mask);
                // transpose load of real data
                #pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float *) local_thread_data)[j] = ((float *) transpose_buffer)[lane_id * transpose_stride + j];
                }

                __syncwarp(active_mask);
                rdft<block_size>((float *) local_thread_data);

                // transpose store of complex data
                #pragma unroll
                for (int j = 0; j < block_size / 2 + 1; j++) {
                    transpose_buffer[lane_id * transpose_stride + j] = local_thread_data[j];
                }

                __syncwarp(active_mask);
                if (lane_id < block_size / 2 + 1) {
                    // transpose load of complex data
                    #pragma unroll
                    for (int j = 0; j < block_size; j++) {
                        local_thread_data[j] = transpose_buffer[j * transpose_stride + lane_id];
                    }

                    __syncwarp((1 << (block_size / 2 + 1)) - 1);
                    dft<block_size>((float *) local_thread_data);
                }
            }

            if (lane_id < block_size / 2 + 1) {
                #pragma unroll
                for (int i = 0; i < block_size; i++) {
                    dft<2 * radius + 1>((float *) &thread_data[i], block_size);
                }
            }

            // frequency_filtering
            if (lane_id < block_size / 2 + 1) {
#if ZERO_MEAN
                float gf;
                if (lane_id == 0) {
                    gf = thread_data[0].x / window_freq[0];
                }
                gf = __shfl_sync((1 << (block_size / 2 + 1)) - 1, gf, 0);
#endif // ZERO_MEAN
                #pragma unroll
                for (int i = 0; i < 2 * radius + 1; i++) {
                    #pragma unroll
                    for (int j = 0; j < block_size; j++) {
                        float2 local_data = thread_data[i * block_size + j];

#if ZERO_MEAN
                        // remove mean
                        float val1 = gf * window_freq[((i * block_size + j) * (block_size / 2 + 1) + lane_id) * 2];
                        float val2 = gf * window_freq[((i * block_size + j) * (block_size / 2 + 1) + lane_id) * 2 + 1];
                        local_data.x -= val1;
                        local_data.y -= val2;
#endif // ZERO_MEAN

                        filter(local_data, lane_id, j, i);

#if ZERO_MEAN
                        // add mean
                        local_data.x += val1;
                        local_data.y += val2;
#endif // ZERO_MEAN

                        thread_data[i * block_size + j] = local_data;
                    }
                }
            }

            // irdft
            if (lane_id < block_size / 2 + 1) {
                #pragma unroll
                for (int i = 0; i < block_size; i++) {
                    idft<2 * radius + 1>((float *) &thread_data[i], block_size);
                }
            }

            #pragma unroll
            for (int i = 0; i < 2 * radius + 1; i++) {
                auto local_thread_data = &thread_data[i * block_size];

                if (lane_id < block_size / 2 + 1) {
                    __syncwarp((1 << (block_size / 2 + 1)) - 1);
                    idft<block_size>((float *) local_thread_data);

                    // transpose store of complex data
                    #pragma unroll
                    for (int j = 0; j < block_size; j++) {
                        transpose_buffer[j * transpose_stride + lane_id] = local_thread_data[j];
                    }
                }

                __syncwarp(active_mask);
                #pragma unroll
                for (int j = 0; j < block_size / 2 + 1; j++) {
                    // transpose load of complex data
                    local_thread_data[j].x = transpose_buffer[lane_id * transpose_stride + j].x;
                    local_thread_data[j].y = transpose_buffer[lane_id * transpose_stride + j].y;
                }

                __syncwarp(active_mask);
                irdft<block_size>((float *) local_thread_data);

                #pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float *) transpose_buffer)[j * transpose_stride + lane_id] = ((float *) local_thread_data)[j == 0 ? j : block_size - j];
                }

                __syncwarp(active_mask);
                auto local_dst = &dstp[(block_id * (2 * radius + 1) + i) * block_size * block_size];
                #pragma unroll
                for (int j = 0; j < block_size; j++) {
                    local_dst[j * block_size + lane_id] = ((float *) transpose_buffer)[lane_id * transpose_stride + j];
                }
            }
        }
    }
}

extern "C"
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
__global__
void col2im(
    TYPE * __restrict__ dst, // shape: (2*radius+1, vertical_size, horizontal_size)
    const float * __restrict__ src, // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    int width,
    int height
) {

    int radius = static_cast<int>(RADIUS);
    int block_size = static_cast<int>(BLOCK_SIZE);
    int block_step = static_cast<int>(BLOCK_STEP);

    // each thread is responsible for a single pixel
    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int pad_x = (horizontal_size - width) / 2;
    int pad_y = (vertical_size - height) / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < pad_y || y >= pad_y + height || x < pad_x || x >= pad_x + width) {
        return ;
    }

    float sum {};

    int i1 = (y - block_size + block_step) / block_step; // i1 is implicitly greater than 0
    int i2 = min(y / block_step, vertical_num - 1);
    int j1 = (x - block_size + block_step) / block_step; // j1 is implicitly greater than 0
    int j2 = min(x / block_step, horizontal_num - 1);

    for (int i = i1; i <= i2; i++) {
        int offset_y = y - i * block_step;
        for (int j = j1; j <= j2; j++) {
            int offset_x = x - j * block_step;
            auto src_offset = (((i * horizontal_num + j) * (2 * radius + 1) + radius) * block_size + offset_y) * block_size + offset_x;
            auto window_offset = (radius * block_size + offset_y) * block_size + offset_x;
            sum += src[src_offset] * window[window_offset];
        }
    }

    dst[(radius * vertical_size + y) * horizontal_size + x] = from_float(sum);
}
)""";

#endif // KERNEL_HPP
