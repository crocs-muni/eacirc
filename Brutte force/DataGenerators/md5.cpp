/*
 **********************************************************************
 ** md5.c                                                            **
 ** RSA Data Security, Inc. MD5 Message Digest Algorithm             **
 ** Created: 2/17/90 RLR                                             **
 ** Revised: 1/91 SRD,AJ,BSK,JT Reference C Version                  **
 **********************************************************************
 */

/*
 **********************************************************************
 ** Copyright (C) 1990, RSA Data Security, Inc. All rights reserved. **
 **                                                                  **
 ** License to copy and use this software is granted provided that   **
 ** it is identified as the "RSA Data Security, Inc. MD5 Message     **
 ** Digest Algorithm" in all material mentioning or referencing this **
 ** software or this function.                                       **
 **                                                                  **
 ** License is also granted to make and use derivative works         **
 ** provided that such works are identified as "derived from the RSA **
 ** Data Security, Inc. MD5 Message Digest Algorithm" in all         **
 ** material mentioning or referencing the derived work.             **
 **                                                                  **
 ** RSA Data Security, Inc. makes no representations concerning      **
 ** either the merchantability of this software or the suitability   **
 ** of this software for any particular purpose.  It is provided "as **
 ** is" without express or implied warranty of any kind.             **
 **                                                                  **
 ** These notices must be retained in any copies of any part of this **
 ** documentation and/or software.                                   **
 **********************************************************************
round reduced modification by Syso 22.10.2016
 */

#include "md5.h"
#include <stdio.h>

/* forward declaration */
static void Transform (UINT4 *buf, UINT4 *in, int Nr = 64);

static unsigned char PADDING[64] = {
  0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

/* F, G and H are basic MD5 functions: selection, majority, parity */
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac) \
  {(a) += F ((b), (c), (d)) + (x) + (UINT4)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define GG(a, b, c, d, x, s, ac) \
  {(a) += G ((b), (c), (d)) + (x) + (UINT4)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define HH(a, b, c, d, x, s, ac) \
  {(a) += H ((b), (c), (d)) + (x) + (UINT4)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define II(a, b, c, d, x, s, ac) \
  {(a) += I ((b), (c), (d)) + (x) + (UINT4)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }

void MD5Init (MD5_CTX* mdContext, int Nr) {
  mdContext->i[0] = mdContext->i[1] = (UINT4)0;


  /* Load magic initialization constants.
   */
  mdContext->buf[0] = (UINT4)0x67452301;
  mdContext->buf[1] = (UINT4)0xefcdab89;
  mdContext->buf[2] = (UINT4)0x98badcfe;
  mdContext->buf[3] = (UINT4)0x10325476;




}

void MD5Update (MD5_CTX *mdContext, unsigned char *inBuf, unsigned int inLen, int Nr) {
  UINT4 in[16];
  int mdi;
  unsigned int i, ii;

  /* compute number of bytes mod 64 */
  mdi = (int)((mdContext->i[0] >> 3) & 0x3F);

  /* update number of bits */
  if ((mdContext->i[0] + ((UINT4)inLen << 3)) < mdContext->i[0])
    mdContext->i[1]++;
  mdContext->i[0] += ((UINT4)inLen << 3);
  mdContext->i[1] += ((UINT4)inLen >> 29);

  while (inLen--) {
    /* add new character to buffer, increment mdi */
    mdContext->in[mdi++] = *inBuf++;

    /* transform if necessary */
    if (mdi == 0x40) {
      for (i = 0, ii = 0; i < 16; i++, ii += 4)
        in[i] = (((UINT4)mdContext->in[ii+3]) << 24) |
                (((UINT4)mdContext->in[ii+2]) << 16) |
                (((UINT4)mdContext->in[ii+1]) << 8) |
                ((UINT4)mdContext->in[ii]);
      Transform (mdContext->buf, in, Nr);
      mdi = 0;
    }
  }
}

void MD5Final (MD5_CTX *mdContext, int Nr) {
  UINT4 in[16];
  int mdi;
  unsigned int i, ii;
  unsigned int padLen;

  /* save number of bits */
  in[14] = mdContext->i[0];
  in[15] = mdContext->i[1];

  /* compute number of bytes mod 64 */
  mdi = (int)((mdContext->i[0] >> 3) & 0x3F);

  /* pad out to 56 mod 64 */
  padLen = (mdi < 56) ? (56 - mdi) : (120 - mdi);
  MD5Update (mdContext, PADDING, padLen, Nr);

  /* append length in bits and transform */
  for (i = 0, ii = 0; i < 14; i++, ii += 4)
    in[i] = (((UINT4)mdContext->in[ii+3]) << 24) |
            (((UINT4)mdContext->in[ii+2]) << 16) |
            (((UINT4)mdContext->in[ii+1]) << 8) |
            ((UINT4)mdContext->in[ii]);
  Transform (mdContext->buf, in,  Nr);

  /* store buffer in digest */
  for (i = 0, ii = 0; i < 4; i++, ii += 4) {
    mdContext->digest[ii] = (unsigned char)(mdContext->buf[i] & 0xFF);
    mdContext->digest[ii+1] =
            (unsigned char)((mdContext->buf[i] >> 8) & 0xFF);
    mdContext->digest[ii+2] =
            (unsigned char)((mdContext->buf[i] >> 16) & 0xFF);
    mdContext->digest[ii+3] =
            (unsigned char)((mdContext->buf[i] >> 24) & 0xFF);
  }
}

/* Basic MD5 step. Transform buf based on in.
 */
static void Transform (UINT4 *buf, UINT4 *in, int Nr ) {
  UINT4 a = buf[0], b = buf[1], c = buf[2], d = buf[3];
  int k = sizeof(a);

  /* Round 1 */
#define S11 7
#define S12 12
#define S13 17
#define S14 22

  FF ( a, b, c, d, in[ 0], S11, 3614090360); /* 1 */ if(Nr == 1)goto label;
  FF ( d, a, b, c, in[ 1], S12, 3905402710); /* 2 */ if(Nr == 2)goto label;
  FF ( c, d, a, b, in[ 2], S13,  606105819); /* 3 */ if(Nr == 3)goto label;
  FF ( b, c, d, a, in[ 3], S14, 3250441966); /* 4 */ if(Nr == 4)goto label;
  FF ( a, b, c, d, in[ 4], S11, 4118548399); /* 5 */ if(Nr == 5)goto label;
  FF ( d, a, b, c, in[ 5], S12, 1200080426); /* 6 */ if(Nr == 6)goto label;
  FF ( c, d, a, b, in[ 6], S13, 2821735955); /* 7 */ if(Nr == 7)goto label;
  FF ( b, c, d, a, in[ 7], S14, 4249261313); /* 8 */ if(Nr == 8)goto label;
  FF ( a, b, c, d, in[ 8], S11, 1770035416); /* 9 */ if(Nr == 9)goto label;
  FF ( d, a, b, c, in[ 9], S12, 2336552879); /* 10 */ if(Nr == 10)goto label;
  FF ( c, d, a, b, in[10], S13, 4294925233); /* 11 */ if(Nr == 11)goto label;
  FF ( b, c, d, a, in[11], S14, 2304563134); /* 12 */ if(Nr == 12)goto label;
  FF ( a, b, c, d, in[12], S11, 1804603682); /* 13 */ if(Nr == 13)goto label;
  FF ( d, a, b, c, in[13], S12, 4254626195); /* 14 */ if(Nr == 14)goto label;
  FF ( c, d, a, b, in[14], S13, 2792965006); /* 15 */ if(Nr == 15)goto label;
  FF ( b, c, d, a, in[15], S14, 1236535329); /* 16 */ if(Nr == 16)goto label;

    /* Round 2 */
#define S21 5
#define S22 9
#define S23 14
#define S24 20
  GG ( a, b, c, d, in[ 1], S21, 4129170786); /* 17 */ if(Nr == 17)goto label;
  GG ( d, a, b, c, in[ 6], S22, 3225465664); /* 18 */ if(Nr == 18)goto label;
  GG ( c, d, a, b, in[11], S23,  643717713); /* 19 */ if(Nr == 19)goto label;
  GG ( b, c, d, a, in[ 0], S24, 3921069994); /* 20 */ if(Nr == 20)goto label;
  GG ( a, b, c, d, in[ 5], S21, 3593408605); /* 21 */ if(Nr == 21)goto label;
  GG ( d, a, b, c, in[10], S22,   38016083); /* 22 */ if(Nr == 22)goto label;
  GG ( c, d, a, b, in[15], S23, 3634488961); /* 23 */ if(Nr == 23)goto label;
  GG ( b, c, d, a, in[ 4], S24, 3889429448); /* 24 */ if(Nr == 24)goto label;
  GG ( a, b, c, d, in[ 9], S21,  568446438); /* 25 */ if(Nr == 25)goto label;
  GG ( d, a, b, c, in[14], S22, 3275163606); /* 26 */ if(Nr == 26)goto label;
  GG ( c, d, a, b, in[ 3], S23, 4107603335); /* 27 */ if(Nr == 27)goto label;
  GG ( b, c, d, a, in[ 8], S24, 1163531501); /* 28 */ if(Nr == 28)goto label;
  GG ( a, b, c, d, in[13], S21, 2850285829); /* 29 */ if(Nr == 29)goto label;
  GG ( d, a, b, c, in[ 2], S22, 4243563512); /* 30 */ if(Nr == 30)goto label;
  GG ( c, d, a, b, in[ 7], S23, 1735328473); /* 31 */ if(Nr == 31)goto label;
  GG ( b, c, d, a, in[12], S24, 2368359562); /* 32 */ if(Nr == 32)goto label;

    /* Round 3 */
#define S31 4
#define S32 11
#define S33 16
#define S34 23
  HH ( a, b, c, d, in[ 5], S31, 4294588738); /* 33 */ if(Nr == 33)goto label;
  HH ( d, a, b, c, in[ 8], S32, 2272392833); /* 34 */ if(Nr == 34)goto label;
  HH ( c, d, a, b, in[11], S33, 1839030562); /* 35 */ if(Nr == 35)goto label;
  HH ( b, c, d, a, in[14], S34, 4259657740); /* 36 */ if(Nr == 36)goto label;
  HH ( a, b, c, d, in[ 1], S31, 2763975236); /* 37 */ if(Nr == 37)goto label;
  HH ( d, a, b, c, in[ 4], S32, 1272893353); /* 38 */ if(Nr == 38)goto label;
  HH ( c, d, a, b, in[ 7], S33, 4139469664); /* 39 */ if(Nr == 39)goto label;
  HH ( b, c, d, a, in[10], S34, 3200236656); /* 40 */ if(Nr == 40)goto label;
  HH ( a, b, c, d, in[13], S31,  681279174); /* 41 */ if(Nr == 41)goto label;
  HH ( d, a, b, c, in[ 0], S32, 3936430074); /* 42 */ if(Nr == 42)goto label;
  HH ( c, d, a, b, in[ 3], S33, 3572445317); /* 43 */ if(Nr == 43)goto label;
  HH ( b, c, d, a, in[ 6], S34,   76029189); /* 44 */ if(Nr == 44)goto label;
  HH ( a, b, c, d, in[ 9], S31, 3654602809); /* 45 */ if(Nr == 45)goto label;
  HH ( d, a, b, c, in[12], S32, 3873151461); /* 46 */ if(Nr == 46)goto label;
  HH ( c, d, a, b, in[15], S33,  530742520); /* 47 */ if(Nr == 47)goto label;
  HH ( b, c, d, a, in[ 2], S34, 3299628645); /* 48 */ if(Nr == 48)goto label;

    /* Round 4 */
#define S41 6
#define S42 10
#define S43 15
#define S44 21
  II ( a, b, c, d, in[ 0], S41, 4096336452); /* 49 */ if(Nr == 49)goto label;
  II ( d, a, b, c, in[ 7], S42, 1126891415); /* 50 */ if(Nr == 50)goto label;
  II ( c, d, a, b, in[14], S43, 2878612391); /* 51 */ if(Nr == 51)goto label;
  II ( b, c, d, a, in[ 5], S44, 4237533241); /* 52 */ if(Nr == 52)goto label;
  II ( a, b, c, d, in[12], S41, 1700485571); /* 53 */ if(Nr == 53)goto label;
  II ( d, a, b, c, in[ 3], S42, 2399980690); /* 54 */ if(Nr == 54)goto label;
  II ( c, d, a, b, in[10], S43, 4293915773); /* 55 */ if(Nr == 55)goto label;
  II ( b, c, d, a, in[ 1], S44, 2240044497); /* 56 */ if(Nr == 56)goto label;
  II ( a, b, c, d, in[ 8], S41, 1873313359); /* 57 */ if(Nr == 57)goto label;
  II ( d, a, b, c, in[15], S42, 4264355552); /* 58 */ if(Nr == 58)goto label;
  II ( c, d, a, b, in[ 6], S43, 2734768916); /* 59 */ if(Nr == 59)goto label;
  II ( b, c, d, a, in[13], S44, 1309151649); /* 60 */ if(Nr == 60)goto label;
  II ( a, b, c, d, in[ 4], S41, 4149444226); /* 61 */ if(Nr == 61)goto label;
  II ( d, a, b, c, in[11], S42, 3174756917); /* 62 */ if(Nr == 62)goto label;
  II ( c, d, a, b, in[ 2], S43,  718787259); /* 63 */ if(Nr == 63)goto label;
  II ( b, c, d, a, in[ 9], S44, 3951481745); /* 64 */ if(Nr == 64)goto label;


  label:
  buf[0] += a;
  buf[1] += b;
  buf[2] += c;
  buf[3] += d;
}
