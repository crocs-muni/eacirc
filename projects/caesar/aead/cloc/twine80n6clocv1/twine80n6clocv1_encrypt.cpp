namespace Twine80n6clocv1_raw {
int numRounds = -1;

#include "twine80n6clocv1_encrypt.h"
#include <string.h>
#include "twine80n6clocv1_api.h"
#include "cloc.h"

int crypto_aead_encrypt(
		unsigned char *c, unsigned long long *clen,
		const unsigned char *m, unsigned long long mlen,
		const unsigned char *ad, unsigned long long adlen,
		const unsigned char *nsec,                          //not relavent to CLOC
		const unsigned char *npub,
		const unsigned char *k
		)
{
    /* set ciphertext length */
    *clen = mlen + CRYPTO_ABYTES;
    
    unsigned char tag[CRYPTO_ABYTES];
    ae_cxt* cxt = ae_allocate();
    if(!cxt)
        return RETURN_MEMORY_FAIL;
    
    /* set key and compute round keys */
    ae_init(cxt, k, CRYPTO_KEYBYTES * 8);
    
    /* process the associated data */
    process_ad(cxt, ad, adlen, npub, CRYPTO_NPUBBYTES);
    
    /* encrypt message */
    ae_encrypt(cxt, (unsigned char*)m, mlen, c, tag, CRYPTO_ABYTES, ENC);
    
    /* copy the tag to the end of ciphertext */
    memcpy(c+mlen, tag, CRYPTO_ABYTES);
	return RETURN_SUCCESS;
}

int crypto_aead_decrypt(
		unsigned char *m,unsigned long long *mlen,
		unsigned char *nsec,                                //not relavent to CLOC
		const unsigned char *c,unsigned long long clen,
		const unsigned char *ad,unsigned long long adlen,
		const unsigned char *npub,
		const unsigned char *k
		)
{
    /* set plaintext length */
    *mlen = clen - CRYPTO_ABYTES;
    
    unsigned char tag[CRYPTO_ABYTES];
    ae_cxt* cxt = ae_allocate();
    if(!cxt)
        return RETURN_MEMORY_FAIL;
    
    /* set key and compute round keys */
    ae_init(cxt, k, CRYPTO_KEYBYTES * 8);
    
    /* process the associated data */
    process_ad(cxt, ad, adlen, npub, CRYPTO_NPUBBYTES);
    
    /* decrypt ciphertext */
    ae_encrypt(cxt, m, *mlen, (unsigned char*)c, tag, CRYPTO_ABYTES, DEC);
    
    /* compare the tag */
    int i;
    for(i = 0; i < CRYPTO_ABYTES; i++)
        if(tag[i] != c[(*mlen) + i]){
            return RETURN_TAG_NO_MATCH;
        }
    
	return RETURN_SUCCESS;
}


} // namespace Twine80n6clocv1_raw
