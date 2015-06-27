# === caesar ===
cmake_minimum_required(VERSION 3.0.2)

include(${PROJECT_SOURCE_DIR}/cmake/eacirc_project.cmake)

set(SOURCES
    CaesarProject.cpp
    CaesarInterface.cpp
    Encryptor.cpp
# === CAESAR algorithms ===
    aead/common/crypto_verify_16.cpp
    aead/common/crypto_core_aes128encrypt.cpp
    aead/common/crypto_core_aes256encrypt.cpp
    aead/acorn128/acorn128_encrypt.cpp
    aead/acorn128/Acorn128.cpp
    aead/aeadaes128ocbtaglen128v1/aeadaes128ocbtaglen128v1_encrypt.cpp
    aead/aeadaes128ocbtaglen128v1/Aeadaes128ocbtaglen128v1.cpp
    aead/aeadaes128ocbtaglen64v1/aeadaes128ocbtaglen64v1_encrypt.cpp
    aead/aeadaes128ocbtaglen64v1/Aeadaes128ocbtaglen64v1.cpp
    aead/aeadaes128ocbtaglen96v1/aeadaes128ocbtaglen96v1_encrypt.cpp
    aead/aeadaes128ocbtaglen96v1/Aeadaes128ocbtaglen96v1.cpp
    aead/aeadaes192ocbtaglen128v1/aeadaes192ocbtaglen128v1_encrypt.cpp
    aead/aeadaes192ocbtaglen128v1/Aeadaes192ocbtaglen128v1.cpp
    aead/aeadaes192ocbtaglen64v1/aeadaes192ocbtaglen64v1_encrypt.cpp
    aead/aeadaes192ocbtaglen64v1/Aeadaes192ocbtaglen64v1.cpp
    aead/aeadaes192ocbtaglen96v1/aeadaes192ocbtaglen96v1_encrypt.cpp
    aead/aeadaes192ocbtaglen96v1/Aeadaes192ocbtaglen96v1.cpp
    aead/aeadaes256ocbtaglen128v1/aeadaes256ocbtaglen128v1_encrypt.cpp
    aead/aeadaes256ocbtaglen128v1/Aeadaes256ocbtaglen128v1.cpp
    aead/aeadaes256ocbtaglen64v1/aeadaes256ocbtaglen64v1_encrypt.cpp
    aead/aeadaes256ocbtaglen64v1/Aeadaes256ocbtaglen64v1.cpp
    aead/aeadaes256ocbtaglen96v1/aeadaes256ocbtaglen96v1_encrypt.cpp
    aead/aeadaes256ocbtaglen96v1/Aeadaes256ocbtaglen96v1.cpp
    aead/aegis128/aegis128_aes.cpp
    aead/aegis128/aegis128_encrypt.cpp
    aead/aegis128/Aegis128.cpp
    aead/aegis128l/aegis128l_aes.cpp
    aead/aegis128l/aegis128l_ecrypt.cpp
    aead/aegis128l/Aegis128l.cpp
    aead/aegis256/aegis256_aes.cpp
    aead/aegis256/aegis256_encrypt.cpp
    aead/aegis256/Aegis256.cpp
#    aead/aes128avalanchev1/aes128avalanchev1_avalanche.cpp \ # compilaation ok, sigsegv while running
#    aead/aes128avalanchev1/aes128avalanchev1_largenumbers.cpp \ # compilaation ok, sigsegv while running
#    aead/aes128avalanchev1/aes128avalanchev1_pcmac.cpp \ # compilaation ok, sigsegv while running
#    aead/aes128avalanchev1/aes128avalanchev1_rmac.cpp \ # compilaation ok, sigsegv while running
#    aead/aes128avalanchev1/Aes128avalanchev1.cpp \ # compilaation ok, sigsegv while running
#    aead/aes128cpfbv1/aes128cpfbv1_encrypt.cpp
#    aead/aes128cpfbv1/Aes128cpfbv1.cpp
    aead/aes128gcmv1/aes128gcmv1_encrypt.cpp
    aead/aes128gcmv1/Aes128gcmv1.cpp
    aead/aes128marble4rv1/aes128marble4rv1_aes_core.cpp
    aead/aes128marble4rv1/aes128marble4rv1_encrypt.cpp
    aead/aes128marble4rv1/aes128marble4rv1_marble.cpp
    aead/aes128marble4rv1/aes128marble4rv1_utils.cpp
    aead/aes128marble4rv1/Aes128marble4rv1.cpp
    aead/aes128n12clocv1/aes128n12clocv1_aes_core.cpp
    aead/aes128n12clocv1/aes128n12clocv1_cloc.cpp
    aead/aes128n12clocv1/aes128n12clocv1_encrypt.cpp
    aead/aes128n12clocv1/aes128n12clocv1_utils.cpp
    aead/aes128n12clocv1/Aes128n12clocv1.cpp
#    aead/aes128n12silcv1/aes128n12silcv1_aes_core.cpp
#    aead/aes128n12silcv1/aes128n12silcv1_encrypt.cpp
#    aead/aes128n12silcv1/aes128n12silcv1_silc.cpp
#    aead/aes128n12silcv1/aes128n12silcv1_utils.cpp
#    aead/aes128n12silcv1/Aes128n12silcv1.cpp
    aead/aes128n8clocv1/aes128n8clocv1_aes_core.cpp
    aead/aes128n8clocv1/aes128n8clocv1_cloc.cpp
    aead/aes128n8clocv1/aes128n8clocv1_encrypt.cpp
    aead/aes128n8clocv1/aes128n8clocv1_utils.cpp
    aead/aes128n8clocv1/Aes128n8clocv1.cpp
#    aead/aes128n8silcv1/aes128n8silcv1_aes_core.cpp
#    aead/aes128n8silcv1/aes128n8silcv1_encrypt.cpp
#    aead/aes128n8silcv1/aes128n8silcv1_silc.cpp
#    aead/aes128n8silcv1/aes128n8silcv1_utils.cpp
#    aead/aes128n8silcv1/Aes128n8silcv1.cpp
    aead/aes128otrpv1/aes128otrpv1_encrypt.cpp
    aead/aes128otrpv1/aes128otrpv1_OTR.cpp
    aead/aes128otrpv1/aes128otrpv1_t-aes_enc_only.cpp
    aead/aes128otrpv1/Aes128otrpv1.cpp
    aead/aes128otrsv1/aes128otrsv1_encrypt.cpp
    aead/aes128otrsv1/aes128otrsv1_OTR.cpp
    aead/aes128otrsv1/aes128otrsv1_t-aes_enc_only.cpp
    aead/aes128otrsv1/Aes128otrsv1.cpp
    aead/aes128poetv1aes128/aes128poetv1aes128_aes.cpp
    aead/aes128poetv1aes128/aes128poetv1aes128_encrypt.cpp
    aead/aes128poetv1aes128/aes128poetv1aes128_poet.cpp
    aead/aes128poetv1aes128/Aes128poetv1aes128.cpp
    aead/aes128poetv1aes4/aes128poetv1aes4_aes.cpp
    aead/aes128poetv1aes4/aes128poetv1aes4_encrypt.cpp
    aead/aes128poetv1aes4/aes128poetv1aes4_poet.cpp
    aead/aes128poetv1aes4/Aes128poetv1aes4.cpp
#    aead/aes192avalanchev1/aes192avalanchev1_avalanche.cpp
#    aead/aes192avalanchev1/aes192avalanchev1_largenumbers.cpp
#    aead/aes192avalanchev1/aes192avalanchev1_pcmac.cpp
#    aead/aes192avalanchev1/aes192avalanchev1_rmac.cpp
#    aead/aes192avalanchev1/Aes192avalanchev1.cpp
#    aead/aes256avalanchev1/aes256avalanchev1_avalanche.cpp
#    aead/aes256avalanchev1/aes256avalanchev1_largenumbers.cpp
#    aead/aes256avalanchev1/aes256avalanchev1_pcmac.cpp
#    aead/aes256avalanchev1/aes256avalanchev1_rmac.cpp
#    aead/aes256avalanchev1/Aes256avalanchev1.cpp
#    aead/aes256cpfbv1/aes256cpfbv1_encrypt.cpp
#    aead/aes256cpfbv1/Aes256cpfbv1.cpp
    aead/aes256gcmv1/aes256gcmv1_encrypt.cpp
    aead/aes256gcmv1/Aes256gcmv1.cpp
    aead/aes256otrpv1/aes256otrpv1_encrypt.cpp
    aead/aes256otrpv1/aes256otrpv1_OTR.cpp
    aead/aes256otrpv1/aes256otrpv1_t-aes_enc_only.cpp
    aead/aes256otrpv1/Aes256otrpv1.cpp
    aead/aes256otrsv1/aes256otrsv1_encrypt.cpp
    aead/aes256otrsv1/aes256otrsv1_OTR.cpp
    aead/aes256otrsv1/aes256otrsv1_t-aes_enc_only.cpp
    aead/aes256otrsv1/Aes256otrsv1.cpp
    aead/aescopav1/aescopav1_aes-core.cpp
    aead/aescopav1/aescopav1_encrypt.cpp
    aead/aescopav1/Aescopav1.cpp
    aead/aesjambuv1/aesjambuv1_aes.cpp
    aead/aesjambuv1/aesjambuv1_encrypt.cpp
    aead/aesjambuv1/Aesjambuv1.cpp
    aead/aezv1/aezv1_aez_ref.cpp
    aead/aezv1/aezv1_encrypt.cpp
    aead/aezv1/aezv1_rijndael-alg-fst.cpp
    aead/aezv1/Aezv1.cpp
    aead/aezv3/aezv3_aez_ref.cpp
    aead/aezv3/aezv3_rijndael-alg-fst.cpp
    aead/aezv3/Aezv3.cpp
    aead/artemia128v1/artemia128v1_encrypt.cpp
    aead/artemia128v1/artemia128v1_jhae_decryption.cpp
    aead/artemia128v1/artemia128v1_jhae_encryption.cpp
    aead/artemia128v1/artemia128v1_jhae_padding.cpp
    aead/artemia128v1/artemia128v1_jhae_permutation.cpp
    aead/artemia128v1/Artemia128v1.cpp
    aead/artemia256v1/artemia256v1_encrypt.cpp
    aead/artemia256v1/artemia256v1_jhae_decryption.cpp
    aead/artemia256v1/artemia256v1_jhae_encryption.cpp
    aead/artemia256v1/artemia256v1_jhae_padding.cpp
    aead/artemia256v1/artemia256v1_jhae_permutation.cpp
    aead/artemia256v1/Artemia256v1.cpp
    aead/ascon128v1/ascon128v1_ascon.cpp
    aead/ascon128v1/Ascon128v1.cpp
    aead/ascon96v1/ascon96v1_ascon.cpp
    aead/ascon96v1/Ascon96v1.cpp
    aead/calicov8/calicov8_calico.cpp
    aead/calicov8/calicov8_encrypt.cpp
    aead/calicov8/Calicov8.cpp
    aead/cba1/cba1_encrypt.cpp
    aead/cba1/Cba1.cpp
    aead/cba10/cba10_encrypt.cpp
    aead/cba10/Cba10.cpp
    aead/cba2/cba2_encrypt.cpp
    aead/cba2/Cba2.cpp
    aead/cba3/cba3_encrypt.cpp
    aead/cba3/Cba3.cpp
    aead/cba4/cba4_encrypt.cpp
    aead/cba4/Cba4.cpp
    aead/cba5/cba5_encrypt.cpp
    aead/cba5/Cba5.cpp
    aead/cba6/cba6_encrypt.cpp
    aead/cba6/Cba6.cpp
    aead/cba7/cba7_encrypt.cpp
    aead/cba7/Cba7.cpp
    aead/cba8/cba8_encrypt.cpp
    aead/cba8/Cba8.cpp
    aead/cba9/cba9_encrypt.cpp
    aead/cba9/Cba9.cpp
    aead/cmcc22v1/cmcc22v1_encrypt.cpp
    aead/cmcc22v1/Cmcc22v1.cpp
    aead/cmcc24v1/cmcc24v1_encrypt.cpp
    aead/cmcc24v1/Cmcc24v1.cpp
    aead/cmcc42v1/cmcc42v1_encrypt.cpp
    aead/cmcc42v1/Cmcc42v1.cpp
    aead/cmcc44v1/cmcc44v1_encrypt.cpp
    aead/cmcc44v1/Cmcc44v1.cpp
    aead/cmcc84v1/cmcc84v1_encrypt.cpp
    aead/cmcc84v1/Cmcc84v1.cpp
    aead/deoxyseq128128v1/deoxyseq128128v1_deoxys.cpp
    aead/deoxyseq128128v1/deoxyseq128128v1_encrypt.cpp
    aead/deoxyseq128128v1/deoxyseq128128v1_tweakableBC.cpp
    aead/deoxyseq128128v1/Deoxyseq128128v1.cpp
    aead/deoxyseq256128v1/deoxyseq256128v1_deoxys.cpp
    aead/deoxyseq256128v1/deoxyseq256128v1_encrypt.cpp
    aead/deoxyseq256128v1/deoxyseq256128v1_tweakableBC.cpp
    aead/deoxyseq256128v1/Deoxyseq256128v1.cpp
    aead/deoxysneq128128v1/deoxysneq128128v1_deoxys.cpp
    aead/deoxysneq128128v1/deoxysneq128128v1_encrypt.cpp
    aead/deoxysneq128128v1/deoxysneq128128v1_tweakableBC.cpp
    aead/deoxysneq128128v1/Deoxysneq128128v1.cpp
    aead/deoxysneq256128v1/deoxysneq256128v1_deoxys.cpp
    aead/deoxysneq256128v1/deoxysneq256128v1_encrypt.cpp
    aead/deoxysneq256128v1/deoxysneq256128v1_tweakableBC.cpp
    aead/deoxysneq256128v1/Deoxysneq256128v1.cpp
    aead/elmd1000v1/elmd1000v1_encrypt.cpp
    aead/elmd1000v1/Elmd1000v1.cpp
    aead/elmd1001v1/elmd1001v1_encrypt.cpp
    aead/elmd1001v1/Elmd1001v1.cpp
    aead/elmd101270v1/elmd101270v1_encrypt.cpp
    aead/elmd101270v1/Elmd101270v1.cpp
    aead/elmd101271v1/elmd101271v1_encrypt.cpp
    aead/elmd101271v1/Elmd101271v1.cpp
    aead/elmd500v1/elmd500v1_encrypt.cpp
    aead/elmd500v1/Elmd500v1.cpp
    aead/elmd501v1/elmd501v1_encrypt.cpp
    aead/elmd501v1/Elmd501v1.cpp
    aead/elmd51270v1/elmd51270v1_encrypt.cpp
    aead/elmd51270v1/Elmd51270v1.cpp
    aead/elmd51271v1/elmd51271v1_encrypt.cpp
    aead/elmd51271v1/Elmd51271v1.cpp
    aead/enchilada128v1/enchilada128v1_aes.cpp
    aead/enchilada128v1/enchilada128v1_auth.cpp
    aead/enchilada128v1/enchilada128v1_chacha.cpp
    aead/enchilada128v1/enchilada128v1_enchilada.cpp
    aead/enchilada128v1/Enchilada128v1.cpp
#    aead/enchilada256v1/enchilada256v1_aescrypt.cpp
#    aead/enchilada256v1/enchilada256v1_aestab.cpp
#    aead/enchilada256v1/enchilada256v1_auth.cpp
#    aead/enchilada256v1/enchilada256v1_chacha.cpp
#    aead/enchilada256v1/enchilada256v1_enchilada256.cpp
#    aead/enchilada256v1/Enchilada256v1.cpp
    aead/hs1sivhiv1/hs1sivhiv1_encrypt.cpp
    aead/hs1sivhiv1/Hs1sivhiv1.cpp
    aead/hs1sivlov1/hs1sivlov1_encrypt.cpp
    aead/hs1sivlov1/Hs1sivlov1.cpp
    aead/hs1sivv1/hs1sivv1_encrypt.cpp
    aead/hs1sivv1/Hs1sivv1.cpp
    aead/icepole128av1/icepole128av1_encrypt.cpp
    aead/icepole128av1/icepole128av1_icepole.cpp
    aead/icepole128av1/Icepole128av1.cpp
    aead/icepole128v1/icepole128v1_encrypt.cpp
    aead/icepole128v1/icepole128v1_icepole.cpp
    aead/icepole128v1/Icepole128v1.cpp
    aead/icepole256av1/icepole256av1_encrypt.cpp
    aead/icepole256av1/icepole256av1_icepole.cpp
    aead/icepole256av1/Icepole256av1.cpp
    aead/ifeedaes128n104v1/ifeedaes128n104v1_encrypt.cpp
    aead/ifeedaes128n104v1/Ifeedaes128n104v1.cpp
    aead/ifeedaes128n96v1/ifeedaes128n96v1_encrypt.cpp
    aead/ifeedaes128n96v1/Ifeedaes128n96v1.cpp
    aead/iscream12v1/iscream12v1_iscream_cipher.cpp
    aead/iscream12v1/iscream12v1_tae.cpp
    aead/iscream12v1/Iscream12v1.cpp
    aead/iscream12v2/iscream12v2_iscream_cipher.cpp
    aead/iscream12v2/iscream12v2_tae.cpp
    aead/iscream12v2/Iscream12v2.cpp
    aead/iscream14v1/iscream14v1_iscream_cipher.cpp
    aead/iscream14v1/iscream14v1_tae.cpp
    aead/iscream14v1/Iscream14v1.cpp
    aead/iscream14v2/iscream14v2_iscream_cipher.cpp
    aead/iscream14v2/iscream14v2_tae.cpp
    aead/iscream14v2/Iscream14v2.cpp
#    aead/joltikeq12864v1/joltikeq12864v1_encrypt.cpp
#    aead/joltikeq12864v1/joltikeq12864v1_joltik.cpp
#    aead/joltikeq12864v1/joltikeq12864v1_tweakableBC.cpp
#    aead/joltikeq12864v1/Joltikeq12864v1.cpp
#    aead/joltikeq6464v1/joltikeq6464v1_encrypt.cpp
#    aead/joltikeq6464v1/joltikeq6464v1_joltik.cpp
#    aead/joltikeq6464v1/joltikeq6464v1_tweakableBC.cpp
#    aead/joltikeq6464v1/Joltikeq6464v1.cpp
#    aead/joltikeq8048v1/joltikeq8048v1_encrypt.cpp
#    aead/joltikeq8048v1/joltikeq8048v1_joltik.cpp
#    aead/joltikeq8048v1/joltikeq8048v1_tweakableBC.cpp
#    aead/joltikeq8048v1/Joltikeq8048v1.cpp
#    aead/joltikeq9696v1/joltikeq9696v1_encrypt.cpp
#    aead/joltikeq9696v1/joltikeq9696v1_joltik.cpp
#    aead/joltikeq9696v1/joltikeq9696v1_tweakableBC.cpp
#    aead/joltikeq9696v1/Joltikeq9696v1.cpp
#    aead/joltikneq12864v1/joltikneq12864v1_encrypt.cpp
#    aead/joltikneq12864v1/joltikneq12864v1_joltik.cpp
#    aead/joltikneq12864v1/joltikneq12864v1_tweakableBC.cpp
#    aead/joltikneq12864v1/Joltikneq12864v1.cpp
#    aead/joltikneq6464v1/joltikneq6464v1_encrypt.cpp
#    aead/joltikneq6464v1/joltikneq6464v1_joltik.cpp
#    aead/joltikneq6464v1/joltikneq6464v1_tweakableBC.cpp
#    aead/joltikneq6464v1/Joltikneq6464v1.cpp
#    aead/joltikneq8048v1/joltikneq8048v1_encrypt.cpp
#    aead/joltikneq8048v1/joltikneq8048v1_joltik.cpp
#    aead/joltikneq8048v1/joltikneq8048v1_tweakableBC.cpp
#    aead/joltikneq8048v1/Joltikneq8048v1.cpp
#    aead/joltikneq9696v1/joltikneq9696v1_encrypt.cpp
#    aead/joltikneq9696v1/joltikneq9696v1_joltik.cpp
#    aead/joltikneq9696v1/joltikneq9696v1_tweakableBC.cpp
#    aead/joltikneq9696v1/Joltikneq9696v1.cpp
#    aead/juliusv1draft/juliusv1draft_aes.cpp
#    aead/juliusv1draft/juliusv1draft_encrypt.cpp
#    aead/juliusv1draft/juliusv1draft_functions.cpp
#    aead/juliusv1draft/Juliusv1draft.cpp
#    aead/ketjejrv1/ketjejrv1_displayIntermediateValues.cpp
#    aead/ketjejrv1/ketjejrv1_encrypt.cpp
#    aead/ketjejrv1/ketjejrv1_KeccakF-200-reference.cpp
#    aead/ketjejrv1/ketjejrv1_KeccakP-200-reference.cpp
#    aead/ketjejrv1/ketjejrv1_Ket.cpp
#    aead/ketjejrv1/ketjejrv1_Ketje.cpp
#    aead/ketjejrv1/Ketjejrv1.cpp
#    aead/ketjesrv1/ketjesrv1_displayIntermediateValues.cpp
#    aead/ketjesrv1/ketjesrv1_encrypt.cpp
#    aead/ketjesrv1/ketjesrv1_KeccakF-400-reference.cpp
#    aead/ketjesrv1/ketjesrv1_KeccakP-400-reference.cpp
#    aead/ketjesrv1/ketjesrv1_Ket.cpp
#    aead/ketjesrv1/ketjesrv1_Ketje.cpp
#    aead/ketjesrv1/Ketjesrv1.cpp
#    aead/kiasueq128v1/kiasueq128v1_encrypt.cpp
#    aead/kiasueq128v1/kiasueq128v1_kiasu.cpp
#    aead/kiasueq128v1/kiasueq128v1_tweakable_aes.cpp
#    aead/kiasueq128v1/Kiasueq128v1.cpp
#    aead/kiasuneq128v1/kiasuneq128v1_encrypt.cpp
#    aead/kiasuneq128v1/kiasuneq128v1_kiasu.cpp
#    aead/kiasuneq128v1/kiasuneq128v1_tweakable_aes.cpp
#    aead/kiasuneq128v1/Kiasuneq128v1.cpp
#    aead/lacv1/lacv1_encrypt.cpp
#    aead/lacv1/Lacv1.cpp
#    aead/lakekeyakv1/lakekeyakv1_displayIntermediateValues.cpp
#    aead/lakekeyakv1/lakekeyakv1_encrypt.cpp
#    aead/lakekeyakv1/lakekeyakv1_KeccakDuplex.cpp
#    aead/lakekeyakv1/lakekeyakv1_KeccakF-1600-reference.cpp
#    aead/lakekeyakv1/lakekeyakv1_KeccakP-1600-12-reference.cpp
#    aead/lakekeyakv1/lakekeyakv1_Keyak.cpp
#    aead/lakekeyakv1/Lakekeyakv1.cpp
#    aead/led80n6silcv1/led80n6silcv1_encrypt.cpp
#    aead/led80n6silcv1/led80n6silcv1_led.cpp
#    aead/led80n6silcv1/led80n6silcv1_silc.cpp
#    aead/led80n6silcv1/led80n6silcv1_utils.cpp
#    aead/led80n6silcv1/Led80n6silcv1.cpp
#    aead/minalpherv1/minalpherv1_encrypt.cpp
#    aead/minalpherv1/Minalpherv1.cpp
#    aead/morus1280128v1/morus1280128v1_encrypt.cpp
#    aead/morus1280128v1/Morus1280128v1.cpp
#    aead/morus1280256v1/morus1280256v1_encrypt.cpp
#    aead/morus1280256v1/Morus1280256v1.cpp
#    aead/morus640128v1/morus640128v1_encrypt.cpp
#    aead/morus640128v1/Morus640128v1.cpp
#    aead/norx3241v1/norx3241v1_caesar.cpp
#    aead/norx3241v1/norx3241v1_norx.cpp
#    aead/norx3241v1/Norx3241v1.cpp
#    aead/norx3261v1/norx3261v1_caesar.cpp
#    aead/norx3261v1/norx3261v1_norx.cpp
#    aead/norx3261v1/Norx3261v1.cpp
#    aead/norx6441v1/norx6441v1_caesar.cpp
#    aead/norx6441v1/norx6441v1_norx.cpp
#    aead/norx6441v1/Norx6441v1.cpp
#    aead/norx6444v1/norx6444v1_caesar.cpp
#    aead/norx6444v1/norx6444v1_norx.cpp
#    aead/norx6444v1/Norx6444v1.cpp
#    aead/norx6461v1/norx6461v1_caesar.cpp
#    aead/norx6461v1/norx6461v1_norx.cpp
#    aead/norx6461v1/Norx6461v1.cpp
#    aead/oceankeyakv1/oceankeyakv1_displayIntermediateValues.cpp
#    aead/oceankeyakv1/oceankeyakv1_encrypt.cpp
#    aead/oceankeyakv1/oceankeyakv1_KeccakDuplex.cpp
#    aead/oceankeyakv1/oceankeyakv1_KeccakF-1600-reference.cpp
#    aead/oceankeyakv1/oceankeyakv1_KeccakP-1600-12-reference.cpp
#    aead/oceankeyakv1/oceankeyakv1_KeccakParallelDuplex.cpp
#    aead/oceankeyakv1/oceankeyakv1_Keyak.cpp
#    aead/oceankeyakv1/oceankeyakv1_SerialFallback.cpp
#    aead/oceankeyakv1/oceankeyakv1_testParallelPaSM.cpp
#    aead/oceankeyakv1/Oceankeyakv1.cpp
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_encrypt.cpp
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_omdsha256.cpp
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_sha256.cpp
#    aead/omdsha256k128n96tau128v1/Omdsha256k128n96tau128v1.cpp
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_encrypt.cpp
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_omdsha256.cpp
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_sha256.cpp
#    aead/omdsha256k128n96tau64v1/Omdsha256k128n96tau64v1.cpp
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_encrypt.cpp
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_omdsha256.cpp
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_sha256.cpp
#    aead/omdsha256k128n96tau96v1/Omdsha256k128n96tau96v1.cpp
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_encrypt.cpp
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_omdsha256.cpp
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_sha256.cpp
#    aead/omdsha256k192n104tau128v1/Omdsha256k192n104tau128v1.cpp
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_encrypt.cpp
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_omdsha256.cpp
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_sha256.cpp
#    aead/omdsha256k256n104tau160v1/Omdsha256k256n104tau160v1.cpp
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_encrypt.cpp
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_omdsha256.cpp
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_sha256.cpp
#    aead/omdsha256k256n248tau256v1/Omdsha256k256n248tau256v1.cpp
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_encrypt.cpp
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_omdsha512.cpp
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_sha512.cpp
#    aead/omdsha512k128n128tau128v1/Omdsha512k128n128tau128v1.cpp
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_encrypt.cpp
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_omdsha512.cpp
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_sha512.cpp
#    aead/omdsha512k256n256tau256v1/Omdsha512k256n256tau256v1.cpp
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_encrypt.cpp
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_omdsha512.cpp
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_sha512.cpp
#    aead/omdsha512k512n256tau256v1/Omdsha512k512n256tau256v1.cpp
#    aead/paeq128/paeq128_encrypt.cpp
#    aead/paeq128/Paeq128.cpp
#    aead/paeq128t/paeq128t_encrypt.cpp
#    aead/paeq128t/Paeq128t.cpp
#    aead/paeq128tnm/paeq128tnm_encrypt.cpp
#    aead/paeq128tnm/Paeq128tnm.cpp
#    aead/paeq160/paeq160_encrypt.cpp
#    aead/paeq160/Paeq160.cpp
#    aead/paeq64/paeq64_encrypt.cpp
#    aead/paeq64/Paeq64.cpp
#    aead/paeq80/paeq80_encrypt.cpp
#    aead/paeq80/Paeq80.cpp
#    aead/pi16cipher096v1/pi16cipher096v1_encrypt.cpp
#    aead/pi16cipher096v1/Pi16cipher096v1.cpp
#    aead/pi16cipher128v1/pi16cipher128v1_encrypt.cpp
#    aead/pi16cipher128v1/Pi16cipher128v1.cpp
#    aead/pi32cipher128v1/pi32cipher128v1_encrypt.cpp
#    aead/pi32cipher128v1/Pi32cipher128v1.cpp
#    aead/pi32cipher256v1/pi32cipher256v1_encrypt.cpp
#    aead/pi32cipher256v1/Pi32cipher256v1.cpp
#    aead/pi64cipher128v1/pi64cipher128v1_encrypt.cpp
#    aead/pi64cipher128v1/Pi64cipher128v1.cpp
#    aead/pi64cipher256v1/pi64cipher256v1_encrypt.cpp
#    aead/pi64cipher256v1/Pi64cipher256v1.cpp
#    aead/pi64cipher256v1oneround/pi64cipher256v1oneround_encrypt.cpp
#    aead/pi64cipher256v1oneround/Pi64cipher256v1oneround.cpp
#    aead/pi64cipher256v1tworounds/pi64cipher256v1tworounds_encrypt.cpp
#    aead/pi64cipher256v1tworounds/Pi64cipher256v1tworounds.cpp
#    aead/polawisv1/polawisv1_decode_POLAWIS.cpp
#    aead/polawisv1/polawisv1_encode_POLAWIS.cpp
#    aead/polawisv1/polawisv1_key_gen_POLAWIS.cpp
#    aead/polawisv1/Polawisv1.cpp
#    aead/ppaev11/ppaev11_encrypt.cpp
#    aead/ppaev11/Ppaev11.cpp
#    aead/present80n6silcv1/present80n6silcv1_encrypt.cpp
#    aead/present80n6silcv1/present80n6silcv1_present.cpp
#    aead/present80n6silcv1/present80n6silcv1_silc.cpp
#    aead/present80n6silcv1/present80n6silcv1_utils.cpp
#    aead/present80n6silcv1/Present80n6silcv1.cpp
#    aead/primatesv1ape120/primatesv1ape120_encrypt.cpp
#    aead/primatesv1ape120/primatesv1ape120_primate.cpp
#    aead/primatesv1ape120/Primatesv1ape120.cpp
#    aead/primatesv1ape80/primatesv1ape80_encrypt.cpp
#    aead/primatesv1ape80/primatesv1ape80_primate.cpp
#    aead/primatesv1ape80/Primatesv1ape80.cpp
#    aead/primatesv1gibbon120/primatesv1gibbon120_encrypt.cpp
#    aead/primatesv1gibbon120/primatesv1gibbon120_primate.cpp
#    aead/primatesv1gibbon120/Primatesv1gibbon120.cpp
#    aead/primatesv1gibbon80/primatesv1gibbon80_encrypt.cpp
#    aead/primatesv1gibbon80/primatesv1gibbon80_primate.cpp
#    aead/primatesv1gibbon80/Primatesv1gibbon80.cpp
#    aead/primatesv1hanuman120/primatesv1hanuman120_encrypt.cpp
#    aead/primatesv1hanuman120/primatesv1hanuman120_primate.cpp
#    aead/primatesv1hanuman120/Primatesv1hanuman120.cpp
#    aead/primatesv1hanuman80/primatesv1hanuman80_encrypt.cpp
#    aead/primatesv1hanuman80/primatesv1hanuman80_primate.cpp
#    aead/primatesv1hanuman80/Primatesv1hanuman80.cpp
#    aead/proest128apev1/proest128apev1_encrypt.cpp
#    aead/proest128apev1/proest128apev1_proest128.cpp
#    aead/proest128apev1/proest128apev1_zerobytes.cpp
#    aead/proest128apev1/Proest128apev1.cpp
#    aead/proest128copav1/proest128copav1_encrypt.cpp
#    aead/proest128copav1/proest128copav1_proest128.cpp
#    aead/proest128copav1/proest128copav1_zerobytes.cpp
#    aead/proest128copav1/Proest128copav1.cpp
#    aead/proest128otrv1/proest128otrv1_encrypt.cpp
#    aead/proest128otrv1/proest128otrv1_proest128.cpp
#    aead/proest128otrv1/proest128otrv1_zerobytes.cpp
#    aead/proest128otrv1/Proest128otrv1.cpp
#    aead/proest256apev1/proest256apev1_encrypt.cpp
#    aead/proest256apev1/proest256apev1_proest256.cpp
#    aead/proest256apev1/proest256apev1_zerobytes.cpp
#    aead/proest256apev1/Proest256apev1.cpp
#    aead/proest256copav1/proest256copav1_encrypt.cpp
#    aead/proest256copav1/proest256copav1_proest256.cpp
#    aead/proest256copav1/proest256copav1_zerobytes.cpp
#    aead/proest256copav1/Proest256copav1.cpp
#    aead/proest256otrv1/proest256otrv1_encrypt.cpp
#    aead/proest256otrv1/proest256otrv1_proest256.cpp
#    aead/proest256otrv1/proest256otrv1_zerobytes.cpp
#    aead/proest256otrv1/Proest256otrv1.cpp
#    aead/raviyoylav1/raviyoylav1_encrypt.cpp
#    aead/raviyoylav1/Raviyoylav1.cpp
#    aead/riverkeyakv1/riverkeyakv1_displayIntermediateValues.cpp
#    aead/riverkeyakv1/riverkeyakv1_encrypt.cpp
#    aead/riverkeyakv1/riverkeyakv1_KeccakDuplex.cpp
#    aead/riverkeyakv1/riverkeyakv1_KeccakF-800-reference.cpp
#    aead/riverkeyakv1/riverkeyakv1_KeccakP-800-12-reference.cpp
#    aead/riverkeyakv1/riverkeyakv1_Keyak.cpp
#    aead/riverkeyakv1/Riverkeyakv1.cpp
#    aead/sablierv1/sablierv1_encrypt.cpp
#    aead/sablierv1/Sablierv1.cpp
#    aead/scream10v1/scream10v1_scream_cipher.cpp
#    aead/scream10v1/scream10v1_tae.cpp
#    aead/scream10v1/Scream10v1.cpp
#    aead/scream10v2/scream10v2_scream_cipher.cpp
#    aead/scream10v2/scream10v2_tae.cpp
#    aead/scream10v2/Scream10v2.cpp
#    aead/scream12v1/scream12v1_scream_cipher.cpp
#    aead/scream12v1/scream12v1_tae.cpp
#    aead/scream12v1/Scream12v1.cpp
#    aead/scream12v2/scream12v2_scream_cipher.cpp
#    aead/scream12v2/scream12v2_tae.cpp
#    aead/scream12v2/Scream12v2.cpp
#    aead/seakeyakv1/seakeyakv1_displayIntermediateValues.cpp
#    aead/seakeyakv1/seakeyakv1_encrypt.cpp
#    aead/seakeyakv1/seakeyakv1_KeccakDuplex.cpp
#    aead/seakeyakv1/seakeyakv1_KeccakF-1600-reference.cpp
#    aead/seakeyakv1/seakeyakv1_KeccakP-1600-12-reference.cpp
#    aead/seakeyakv1/seakeyakv1_KeccakParallelDuplex.cpp
#    aead/seakeyakv1/seakeyakv1_Keyak.cpp
#    aead/seakeyakv1/seakeyakv1_SerialFallback.cpp
#    aead/seakeyakv1/seakeyakv1_testParallelPaSM.cpp
#    aead/seakeyakv1/Seakeyakv1.cpp
#    aead/shellaes128v1d4n64/shellaes128v1d4n64_aes.cpp
#    aead/shellaes128v1d4n64/shellaes128v1d4n64_aesReduced.cpp
#    aead/shellaes128v1d4n64/shellaes128v1d4n64_encrypt.cpp
#    aead/shellaes128v1d4n64/shellaes128v1d4n64_shellaes.cpp
#    aead/shellaes128v1d4n64/Shellaes128v1d4n64.cpp
#    aead/shellaes128v1d4n80/shellaes128v1d4n80_aes.cpp
#    aead/shellaes128v1d4n80/shellaes128v1d4n80_aesReduced.cpp
#    aead/shellaes128v1d4n80/shellaes128v1d4n80_encrypt.cpp
#    aead/shellaes128v1d4n80/shellaes128v1d4n80_shellaes.cpp
#    aead/shellaes128v1d4n80/Shellaes128v1d4n80.cpp
#    aead/shellaes128v1d5n64/shellaes128v1d5n64_aes.cpp
#    aead/shellaes128v1d5n64/shellaes128v1d5n64_aesReduced.cpp
#    aead/shellaes128v1d5n64/shellaes128v1d5n64_encrypt.cpp
#    aead/shellaes128v1d5n64/shellaes128v1d5n64_shellaes.cpp
#    aead/shellaes128v1d5n64/Shellaes128v1d5n64.cpp
#    aead/shellaes128v1d5n80/shellaes128v1d5n80_aes.cpp
#    aead/shellaes128v1d5n80/shellaes128v1d5n80_aesReduced.cpp
#    aead/shellaes128v1d5n80/shellaes128v1d5n80_encrypt.cpp
#    aead/shellaes128v1d5n80/shellaes128v1d5n80_shellaes.cpp
#    aead/shellaes128v1d5n80/Shellaes128v1d5n80.cpp
#    aead/shellaes128v1d6n64/shellaes128v1d6n64_aes.cpp
#    aead/shellaes128v1d6n64/shellaes128v1d6n64_aesReduced.cpp
#    aead/shellaes128v1d6n64/shellaes128v1d6n64_encrypt.cpp
#    aead/shellaes128v1d6n64/shellaes128v1d6n64_shellaes.cpp
#    aead/shellaes128v1d6n64/Shellaes128v1d6n64.cpp
#    aead/shellaes128v1d6n80/shellaes128v1d6n80_aes.cpp
#    aead/shellaes128v1d6n80/shellaes128v1d6n80_aesReduced.cpp
#    aead/shellaes128v1d6n80/shellaes128v1d6n80_encrypt.cpp
#    aead/shellaes128v1d6n80/shellaes128v1d6n80_shellaes.cpp
#    aead/shellaes128v1d6n80/Shellaes128v1d6n80.cpp
#    aead/shellaes128v1d7n64/shellaes128v1d7n64_aes.cpp
#    aead/shellaes128v1d7n64/shellaes128v1d7n64_aesReduced.cpp
#    aead/shellaes128v1d7n64/shellaes128v1d7n64_encrypt.cpp
#    aead/shellaes128v1d7n64/shellaes128v1d7n64_shellaes.cpp
#    aead/shellaes128v1d7n64/Shellaes128v1d7n64.cpp
#    aead/shellaes128v1d7n80/shellaes128v1d7n80_aes.cpp
#    aead/shellaes128v1d7n80/shellaes128v1d7n80_aesReduced.cpp
#    aead/shellaes128v1d7n80/shellaes128v1d7n80_encrypt.cpp
#    aead/shellaes128v1d7n80/shellaes128v1d7n80_shellaes.cpp
#    aead/shellaes128v1d7n80/Shellaes128v1d7n80.cpp
#    aead/shellaes128v1d8n64/shellaes128v1d8n64_aes.cpp
#    aead/shellaes128v1d8n64/shellaes128v1d8n64_aesReduced.cpp
#    aead/shellaes128v1d8n64/shellaes128v1d8n64_encrypt.cpp
#    aead/shellaes128v1d8n64/shellaes128v1d8n64_shellaes.cpp
#    aead/shellaes128v1d8n64/Shellaes128v1d8n64.cpp
#    aead/shellaes128v1d8n80/shellaes128v1d8n80_aes.cpp
#    aead/shellaes128v1d8n80/shellaes128v1d8n80_aesReduced.cpp
#    aead/shellaes128v1d8n80/shellaes128v1d8n80_encrypt.cpp
#    aead/shellaes128v1d8n80/shellaes128v1d8n80_shellaes.cpp
#    aead/shellaes128v1d8n80/Shellaes128v1d8n80.cpp
#    aead/silverv1/silverv1_encrypt.cpp
#    aead/silverv1/silverv1_rijndaelEndianNeutral.cpp
#    aead/silverv1/Silverv1.cpp
#    aead/stribob192r1/stribob192r1_encrypt.cpp
#    aead/stribob192r1/stribob192r1_sbob_pi64.cpp
#    aead/stribob192r1/stribob192r1_sbob_tab64.cpp
#    aead/stribob192r1/stribob192r1_stribob.cpp
#    aead/stribob192r1/Stribob192r1.cpp
#    aead/tiaoxinv1/tiaoxinv1_aes_round.cpp
#    aead/tiaoxinv1/tiaoxinv1_encrypt.cpp
#    aead/tiaoxinv1/tiaoxinv1_tiaoxin-reference.cpp
#    aead/tiaoxinv1/Tiaoxinv1.cpp
#    aead/trivia0v1/trivia0v1_encrypt.cpp
#    aead/trivia0v1/Trivia0v1.cpp
#    aead/trivia128v1/trivia128v1_encrypt.cpp
#    aead/trivia128v1/Trivia128v1.cpp
#    aead/twine80n6clocv1/twine80n6clocv1_cloc.cpp
#    aead/twine80n6clocv1/twine80n6clocv1_encrypt.cpp
#    aead/twine80n6clocv1/twine80n6clocv1_twine.cpp
#    aead/twine80n6clocv1/twine80n6clocv1_utils.cpp
#    aead/twine80n6clocv1/Twine80n6clocv1.cpp
#    aead/wheeshtv1mr3fr1t128/wheeshtv1mr3fr1t128_encrypt.cpp
#    aead/wheeshtv1mr3fr1t128/Wheeshtv1mr3fr1t128.cpp
#    aead/wheeshtv1mr3fr1t256/wheeshtv1mr3fr1t256_encrypt.cpp
#    aead/wheeshtv1mr3fr1t256/Wheeshtv1mr3fr1t256.cpp
#    aead/wheeshtv1mr3fr3t256/wheeshtv1mr3fr3t256_encrypt.cpp
#    aead/wheeshtv1mr3fr3t256/Wheeshtv1mr3fr3t256.cpp
#    aead/wheeshtv1mr5fr7t256/wheeshtv1mr5fr7t256_encrypt.cpp
#    aead/wheeshtv1mr5fr7t256/Wheeshtv1mr5fr7t256.cpp
#    aead/yaes128v2/yaes128v2_aes-128.cpp
#    aead/yaes128v2/yaes128v2_encrypt.cpp
#    aead/yaes128v2/Yaes128v2.cpp
)

set(HEADERS
    CaesarConstants.h
    CaesarProject.h
    CaesarInterface.h
    Encryptor.h
# === CAESAR algorithms ===
    aead/common/api.h
    aead/aead.h
    aead/acorn128/acorn128_api.h
    aead/acorn128/acorn128_encrypt.h
    aead/acorn128/Acorn128.h
    aead/aeadaes128ocbtaglen128v1/aeadaes128ocbtaglen128v1_api.h
    aead/aeadaes128ocbtaglen128v1/aeadaes128ocbtaglen128v1_encrypt.h
    aead/aeadaes128ocbtaglen128v1/Aeadaes128ocbtaglen128v1.h
    aead/aeadaes128ocbtaglen64v1/aeadaes128ocbtaglen64v1_api.h
    aead/aeadaes128ocbtaglen64v1/aeadaes128ocbtaglen64v1_encrypt.h
    aead/aeadaes128ocbtaglen64v1/Aeadaes128ocbtaglen64v1.h
    aead/aeadaes128ocbtaglen96v1/aeadaes128ocbtaglen96v1_api.h
    aead/aeadaes128ocbtaglen96v1/aeadaes128ocbtaglen96v1_encrypt.h
    aead/aeadaes128ocbtaglen96v1/Aeadaes128ocbtaglen96v1.h
    aead/aeadaes192ocbtaglen128v1/aeadaes192ocbtaglen128v1_api.h
    aead/aeadaes192ocbtaglen128v1/aeadaes192ocbtaglen128v1_encrypt.h
    aead/aeadaes192ocbtaglen128v1/Aeadaes192ocbtaglen128v1.h
    aead/aeadaes192ocbtaglen64v1/aeadaes192ocbtaglen64v1_api.h
    aead/aeadaes192ocbtaglen64v1/aeadaes192ocbtaglen64v1_encrypt.h
    aead/aeadaes192ocbtaglen64v1/Aeadaes192ocbtaglen64v1.h
    aead/aeadaes192ocbtaglen96v1/aeadaes192ocbtaglen96v1_api.h
    aead/aeadaes192ocbtaglen96v1/aeadaes192ocbtaglen96v1_encrypt.h
    aead/aeadaes192ocbtaglen96v1/Aeadaes192ocbtaglen96v1.h
    aead/aeadaes256ocbtaglen128v1/aeadaes256ocbtaglen128v1_api.h
    aead/aeadaes256ocbtaglen128v1/aeadaes256ocbtaglen128v1_encrypt.h
    aead/aeadaes256ocbtaglen128v1/Aeadaes256ocbtaglen128v1.h
    aead/aeadaes256ocbtaglen64v1/aeadaes256ocbtaglen64v1_api.h
    aead/aeadaes256ocbtaglen64v1/aeadaes256ocbtaglen64v1_encrypt.h
    aead/aeadaes256ocbtaglen64v1/Aeadaes256ocbtaglen64v1.h
    aead/aeadaes256ocbtaglen96v1/aeadaes256ocbtaglen96v1_api.h
    aead/aeadaes256ocbtaglen96v1/aeadaes256ocbtaglen96v1_encrypt.h
    aead/aeadaes256ocbtaglen96v1/Aeadaes256ocbtaglen96v1.h
    aead/aegis128/aegis128_api.h
    aead/aegis128/aegis128_encrypt.h
    aead/aegis128/Aegis128.h
    aead/aegis128l/aegis128l_api.h
    aead/aegis128l/aegis128l_encrypt.h
    aead/aegis128l/Aegis128l.h
    aead/aegis256/aegis256_api.h
    aead/aegis256/aegis256_encrypt.h
    aead/aegis256/Aegis256.h
#    aead/aes128avalanchev1/aes128avalanchev1_api.h \ # compilaation ok, sigsegv while running
#    aead/aes128avalanchev1/aes128avalanchev1_avalanche.h \ # compilaation ok, sigsegv while running
#    aead/aes128avalanchev1/aes128avalanchev1_encrypt.h \ # compilaation ok, sigsegv while running
#    aead/aes128avalanchev1/Aes128avalanchev1.h \ # compilaation ok, sigsegv while running
#    aead/aes128cpfbv1/aes128cpfbv1_api.h
#    aead/aes128cpfbv1/aes128cpfbv1_encrypt.h
#    aead/aes128cpfbv1/Aes128cpfbv1.h
    aead/aes128gcmv1/aes128gcmv1_api.h
    aead/aes128gcmv1/aes128gcmv1_encrypt.h
    aead/aes128gcmv1/Aes128gcmv1.h
    aead/aes128marble4rv1/aes128marble4rv1_aes.h
    aead/aes128marble4rv1/aes128marble4rv1_aes_locl.h
    aead/aes128marble4rv1/aes128marble4rv1_api.h
    aead/aes128marble4rv1/aes128marble4rv1_marble.h
    aead/aes128marble4rv1/aes128marble4rv1_encrypt.h
    aead/aes128marble4rv1/Aes128marble4rv1.h
    aead/aes128n12clocv1/aes128n12clocv1_aes.h
    aead/aes128n12clocv1/aes128n12clocv1_aes_locl.h
    aead/aes128n12clocv1/aes128n12clocv1_api.h
    aead/aes128n12clocv1/aes128n12clocv1_cloc.h
    aead/aes128n12clocv1/aes128n12clocv1_encrypt.h
    aead/aes128n12clocv1/Aes128n12clocv1.h
#    aead/aes128n12silcv1/aes128n12silcv1_aes.h
#    aead/aes128n12silcv1/aes128n12silcv1_aes_locl.h
#    aead/aes128n12silcv1/aes128n12silcv1_api.h
#    aead/aes128n12silcv1/aes128n12silcv1_silc.h
#    aead/aes128n12silcv1/aes128n12silcv1_encrypt.h
#    aead/aes128n12silcv1/Aes128n12silcv1.h
    aead/aes128n8clocv1/aes128n8clocv1_aes.h
    aead/aes128n8clocv1/aes128n8clocv1_aes_locl.h
    aead/aes128n8clocv1/aes128n8clocv1_api.h
    aead/aes128n8clocv1/aes128n8clocv1_cloc.h
    aead/aes128n8clocv1/aes128n8clocv1_encrypt.h
    aead/aes128n8clocv1/Aes128n8clocv1.h
#    aead/aes128n8silcv1/aes128n8silcv1_aes.h
#    aead/aes128n8silcv1/aes128n8silcv1_aes_locl.h
#    aead/aes128n8silcv1/aes128n8silcv1_api.h
#    aead/aes128n8silcv1/aes128n8silcv1_silc.h
#    aead/aes128n8silcv1/aes128n8silcv1_encrypt.h
#    aead/aes128n8silcv1/Aes128n8silcv1.h
    aead/aes128otrpv1/aes128otrpv1_api.h
    aead/aes128otrpv1/aes128otrpv1_OTR.h
    aead/aes128otrpv1/aes128otrpv1_t-aes_define.h
    aead/aes128otrpv1/aes128otrpv1_t-aes_table_enc_only.h
    aead/aes128otrpv1/aes128otrpv1_encrypt.h
    aead/aes128otrpv1/Aes128otrpv1.h
    aead/aes128otrsv1/aes128otrsv1_api.h
    aead/aes128otrsv1/aes128otrsv1_OTR.h
    aead/aes128otrsv1/aes128otrsv1_t-aes_define.h
    aead/aes128otrsv1/aes128otrsv1_t-aes_table_enc_only.h
    aead/aes128otrsv1/aes128otrsv1_encrypt.h
    aead/aes128otrsv1/Aes128otrsv1.h
    aead/aes128poetv1aes128/aes128poetv1aes128_aes.h
    aead/aes128poetv1aes128/aes128poetv1aes128_api.h
    aead/aes128poetv1aes128/aes128poetv1aes128_gf_mul.h
    aead/aes128poetv1aes128/aes128poetv1aes128_poet.h
    aead/aes128poetv1aes128/aes128poetv1aes128_encrypt.h
    aead/aes128poetv1aes128/Aes128poetv1aes128.h
    aead/aes128poetv1aes4/aes128poetv1aes4_aes.h
    aead/aes128poetv1aes4/aes128poetv1aes4_api.h
    aead/aes128poetv1aes4/aes128poetv1aes4_gf_mul.h
    aead/aes128poetv1aes4/aes128poetv1aes4_poet.h
    aead/aes128poetv1aes4/aes128poetv1aes4_encrypt.h
    aead/aes128poetv1aes4/Aes128poetv1aes4.h
 #    aead/aes192avalanchev1/aes192avalanchev1_api.h
 #    aead/aes192avalanchev1/aes192avalanchev1_avalanche.h
 #    aead/aes192avalanchev1/aes192avalanchev1_encrypt.h
 #    aead/aes192avalanchev1/Aes192avalanchev1.h
 #    aead/aes256avalanchev1/aes256avalanchev1_api.h
 #    aead/aes256avalanchev1/aes256avalanchev1_avalanche.h
 #    aead/aes256avalanchev1/aes256avalanchev1_encrypt.h
 #    aead/aes256avalanchev1/Aes256avalanchev1.h
 #    aead/aes256cpfbv1/aes256cpfbv1_api.h
 #    aead/aes256cpfbv1/aes256cpfbv1_encrypt.h
 #    aead/aes256cpfbv1/Aes256cpfbv1.h
    aead/aes256gcmv1/aes256gcmv1_api.h
    aead/aes256gcmv1/aes256gcmv1_encrypt.h
    aead/aes256gcmv1/Aes256gcmv1.h
    aead/aes256otrpv1/aes256otrpv1_api.h
    aead/aes256otrpv1/aes256otrpv1_OTR.h
    aead/aes256otrpv1/aes256otrpv1_t-aes_define.h
    aead/aes256otrpv1/aes256otrpv1_t-aes_table_enc_only.h
    aead/aes256otrpv1/aes256otrpv1_encrypt.h
    aead/aes256otrpv1/Aes256otrpv1.h
    aead/aes256otrsv1/aes256otrsv1_api.h
    aead/aes256otrsv1/aes256otrsv1_OTR.h
    aead/aes256otrsv1/aes256otrsv1_t-aes_define.h
    aead/aes256otrsv1/aes256otrsv1_t-aes_table_enc_only.h
    aead/aes256otrsv1/aes256otrsv1_encrypt.h
    aead/aes256otrsv1/Aes256otrsv1.h
    aead/aescopav1/aescopav1_aes-core.h
    aead/aescopav1/aescopav1_api.h
    aead/aescopav1/aescopav1_encrypt.h
    aead/aescopav1/Aescopav1.h
    aead/aesjambuv1/aesjambuv1_aes.h
    aead/aesjambuv1/aesjambuv1_api.h
    aead/aesjambuv1/aesjambuv1_encrypt.h
    aead/aesjambuv1/Aesjambuv1.h
    aead/aezv1/aezv1_api.h
    aead/aezv1/aezv1_rijndael-alg-fst.h
    aead/aezv1/aezv1_encrypt.h
    aead/aezv1/Aezv1.h
    aead/aezv3/aezv3_api.h
    aead/aezv3/aezv3_rijndael-alg-fst.h
    aead/aezv3/aezv3_encrypt.h
    aead/aezv3/Aezv3.h
    aead/artemia128v1/artemia128v1_api.h
    aead/artemia128v1/artemia128v1_jhae.h
    aead/artemia128v1/artemia128v1_encrypt.h
    aead/artemia128v1/Artemia128v1.h
    aead/artemia256v1/artemia256v1_api.h
    aead/artemia256v1/artemia256v1_jhae.h
    aead/artemia256v1/artemia256v1_encrypt.h
    aead/artemia256v1/Artemia256v1.h
    aead/ascon128v1/ascon128v1_api.h
    aead/ascon128v1/ascon128v1_encrypt.h
    aead/ascon128v1/Ascon128v1.h
    aead/ascon96v1/ascon96v1_api.h
    aead/ascon96v1/ascon96v1_encrypt.h
    aead/ascon96v1/Ascon96v1.h
    aead/calicov8/calicov8_api.h
    aead/calicov8/calicov8_calico.h
    aead/calicov8/calicov8_encrypt.h
    aead/calicov8/Calicov8.h
    aead/cba1/cba1_api.h
    aead/cba1/cba1_encrypt.h
    aead/cba1/Cba1.h
    aead/cba10/cba10_api.h
    aead/cba10/cba10_encrypt.h
    aead/cba10/Cba10.h
    aead/cba2/cba2_api.h
    aead/cba2/cba2_encrypt.h
    aead/cba2/Cba2.h
    aead/cba3/cba3_api.h
    aead/cba3/cba3_encrypt.h
    aead/cba3/Cba3.h
    aead/cba4/cba4_api.h
    aead/cba4/cba4_encrypt.h
    aead/cba4/Cba4.h
    aead/cba5/cba5_api.h
    aead/cba5/cba5_encrypt.h
    aead/cba5/Cba5.h
    aead/cba6/cba6_api.h
    aead/cba6/cba6_encrypt.h
    aead/cba6/Cba6.h
    aead/cba7/cba7_api.h
    aead/cba7/cba7_encrypt.h
    aead/cba7/Cba7.h
    aead/cba8/cba8_api.h
    aead/cba8/cba8_encrypt.h
    aead/cba8/Cba8.h
    aead/cba9/cba9_api.h
    aead/cba9/cba9_encrypt.h
    aead/cba9/Cba9.h
    aead/cmcc22v1/cmcc22v1_api.h
    aead/cmcc22v1/cmcc22v1_encrypt.h
    aead/cmcc22v1/Cmcc22v1.h
    aead/cmcc24v1/cmcc24v1_api.h
    aead/cmcc24v1/cmcc24v1_encrypt.h
    aead/cmcc24v1/Cmcc24v1.h
    aead/cmcc42v1/cmcc42v1_api.h
    aead/cmcc42v1/cmcc42v1_encrypt.h
    aead/cmcc42v1/Cmcc42v1.h
    aead/cmcc44v1/cmcc44v1_api.h
    aead/cmcc44v1/cmcc44v1_encrypt.h
    aead/cmcc44v1/Cmcc44v1.h
    aead/cmcc84v1/cmcc84v1_api.h
    aead/cmcc84v1/cmcc84v1_encrypt.h
    aead/cmcc84v1/Cmcc84v1.h
    aead/deoxyseq128128v1/deoxyseq128128v1_api.h
    aead/deoxyseq128128v1/deoxyseq128128v1_deoxys.h
    aead/deoxyseq128128v1/deoxyseq128128v1_tweakableBC.h
    aead/deoxyseq128128v1/deoxyseq128128v1_encrypt.h
    aead/deoxyseq128128v1/Deoxyseq128128v1.h
    aead/deoxyseq256128v1/deoxyseq256128v1_api.h
    aead/deoxyseq256128v1/deoxyseq256128v1_deoxys.h
    aead/deoxyseq256128v1/deoxyseq256128v1_tweakableBC.h
    aead/deoxyseq256128v1/deoxyseq256128v1_encrypt.h
    aead/deoxyseq256128v1/Deoxyseq256128v1.h
    aead/deoxysneq128128v1/deoxysneq128128v1_api.h
    aead/deoxysneq128128v1/deoxysneq128128v1_deoxys.h
    aead/deoxysneq128128v1/deoxysneq128128v1_tweakableBC.h
    aead/deoxysneq128128v1/deoxysneq128128v1_encrypt.h
    aead/deoxysneq128128v1/Deoxysneq128128v1.h
    aead/deoxysneq256128v1/deoxysneq256128v1_api.h
    aead/deoxysneq256128v1/deoxysneq256128v1_deoxys.h
    aead/deoxysneq256128v1/deoxysneq256128v1_tweakableBC.h
    aead/deoxysneq256128v1/deoxysneq256128v1_encrypt.h
    aead/deoxysneq256128v1/Deoxysneq256128v1.h
    aead/elmd1000v1/elmd1000v1_aes_round_5.h
    aead/elmd1000v1/elmd1000v1_api.h
    aead/elmd1000v1/elmd1000v1_module.h
    aead/elmd1000v1/elmd1000v1_encrypt.h
    aead/elmd1000v1/Elmd1000v1.h
    aead/elmd1001v1/elmd1001v1_aes_round_5.h
    aead/elmd1001v1/elmd1001v1_api.h
    aead/elmd1001v1/elmd1001v1_module.h
    aead/elmd1001v1/elmd1001v1_encrypt.h
    aead/elmd1001v1/Elmd1001v1.h
    aead/elmd101270v1/elmd101270v1_aes_round_5.h
    aead/elmd101270v1/elmd101270v1_api.h
    aead/elmd101270v1/elmd101270v1_module.h
    aead/elmd101270v1/elmd101270v1_encrypt.h
    aead/elmd101270v1/Elmd101270v1.h
    aead/elmd101271v1/elmd101271v1_aes_round_5.h
    aead/elmd101271v1/elmd101271v1_api.h
    aead/elmd101271v1/elmd101271v1_module.h
    aead/elmd101271v1/elmd101271v1_encrypt.h
    aead/elmd101271v1/Elmd101271v1.h
    aead/elmd500v1/elmd500v1_aes_round_5.h
    aead/elmd500v1/elmd500v1_api.h
    aead/elmd500v1/elmd500v1_module.h
    aead/elmd500v1/elmd500v1_encrypt.h
    aead/elmd500v1/Elmd500v1.h
    aead/elmd501v1/elmd501v1_aes_round_5.h
    aead/elmd501v1/elmd501v1_api.h
    aead/elmd501v1/elmd501v1_module.h
    aead/elmd501v1/elmd501v1_encrypt.h
    aead/elmd501v1/Elmd501v1.h
    aead/elmd51270v1/elmd51270v1_aes_round_5.h
    aead/elmd51270v1/elmd51270v1_api.h
    aead/elmd51270v1/elmd51270v1_module.h
    aead/elmd51270v1/elmd51270v1_encrypt.h
    aead/elmd51270v1/Elmd51270v1.h
    aead/elmd51271v1/elmd51271v1_aes_round_5.h
    aead/elmd51271v1/elmd51271v1_api.h
    aead/elmd51271v1/elmd51271v1_module.h
    aead/elmd51271v1/elmd51271v1_encrypt.h
    aead/elmd51271v1/Elmd51271v1.h
    aead/enchilada128v1/enchilada128v1_api.h
    aead/enchilada128v1/enchilada128v1_enchilada.h
    aead/enchilada128v1/enchilada128v1_encrypt.h
    aead/enchilada128v1/Enchilada128v1.h
#    aead/enchilada256v1/enchilada256v1_aes.h
#    aead/enchilada256v1/enchilada256v1_aesopt.h
#    aead/enchilada256v1/enchilada256v1_api.h
#    aead/enchilada256v1/enchilada256v1_enchilada.h
#    aead/enchilada256v1/enchilada256v1_encrypt.h
#    aead/enchilada256v1/Enchilada256v1.h
    aead/hs1sivhiv1/hs1sivhiv1_api.h
    aead/hs1sivhiv1/hs1sivhiv1_encrypt.h
    aead/hs1sivhiv1/Hs1sivhiv1.h
    aead/hs1sivlov1/hs1sivlov1_api.h
    aead/hs1sivlov1/hs1sivlov1_encrypt.h
    aead/hs1sivlov1/Hs1sivlov1.h
    aead/hs1sivv1/hs1sivv1_api.h
    aead/hs1sivv1/hs1sivv1_encrypt.h
    aead/hs1sivv1/Hs1sivv1.h
    aead/icepole128av1/icepole128av1_api.h
    aead/icepole128av1/icepole128av1_icepole.h
    aead/icepole128av1/icepole128av1_encrypt.h
    aead/icepole128av1/Icepole128av1.h
    aead/icepole128v1/icepole128v1_api.h
    aead/icepole128v1/icepole128v1_icepole.h
    aead/icepole128v1/icepole128v1_encrypt.h
    aead/icepole128v1/Icepole128v1.h
    aead/icepole256av1/icepole256av1_api.h
    aead/icepole256av1/icepole256av1_icepole.h
    aead/icepole256av1/icepole256av1_encrypt.h
    aead/icepole256av1/Icepole256av1.h
    aead/ifeedaes128n104v1/ifeedaes128n104v1_api.h
    aead/ifeedaes128n104v1/ifeedaes128n104v1_encrypt.h
    aead/ifeedaes128n104v1/Ifeedaes128n104v1.h
    aead/ifeedaes128n96v1/ifeedaes128n96v1_api.h
    aead/ifeedaes128n96v1/ifeedaes128n96v1_encrypt.h
    aead/ifeedaes128n96v1/Ifeedaes128n96v1.h
    aead/iscream12v1/iscream12v1_api.h
    aead/iscream12v1/iscream12v1_lbox.h
    aead/iscream12v1/iscream12v1_params.h
    aead/iscream12v1/iscream12v1_sbox.h
    aead/iscream12v1/iscream12v1_encrypt.h
    aead/iscream12v1/Iscream12v1.h
    aead/iscream12v2/iscream12v2_api.h
    aead/iscream12v2/iscream12v2_lbox.h
    aead/iscream12v2/iscream12v2_params.h
    aead/iscream12v2/iscream12v2_sbox.h
    aead/iscream12v2/iscream12v2_encrypt.h
    aead/iscream12v2/Iscream12v2.h
    aead/iscream14v1/iscream14v1_api.h
    aead/iscream14v1/iscream14v1_lbox.h
    aead/iscream14v1/iscream14v1_params.h
    aead/iscream14v1/iscream14v1_sbox.h
    aead/iscream14v1/iscream14v1_encrypt.h
    aead/iscream14v1/Iscream14v1.h
    aead/iscream14v2/iscream14v2_api.h
    aead/iscream14v2/iscream14v2_lbox.h
    aead/iscream14v2/iscream14v2_params.h
    aead/iscream14v2/iscream14v2_sbox.h
    aead/iscream14v2/iscream14v2_encrypt.h
    aead/iscream14v2/Iscream14v2.h
#    aead/joltikeq12864v1/joltikeq12864v1_api.h
#    aead/joltikeq12864v1/joltikeq12864v1_crypto_aead.h
#    aead/joltikeq12864v1/joltikeq12864v1_joltik.h
#    aead/joltikeq12864v1/joltikeq12864v1_tweakableBC.h
#    aead/joltikeq12864v1/joltikeq12864v1_encrypt.h
#    aead/joltikeq12864v1/Joltikeq12864v1.h
#    aead/joltikeq6464v1/joltikeq6464v1_api.h
#    aead/joltikeq6464v1/joltikeq6464v1_crypto_aead.h
#    aead/joltikeq6464v1/joltikeq6464v1_joltik.h
#    aead/joltikeq6464v1/joltikeq6464v1_tweakableBC.h
#    aead/joltikeq6464v1/joltikeq6464v1_encrypt.h
#    aead/joltikeq6464v1/Joltikeq6464v1.h
#    aead/joltikeq8048v1/joltikeq8048v1_api.h
#    aead/joltikeq8048v1/joltikeq8048v1_crypto_aead.h
#    aead/joltikeq8048v1/joltikeq8048v1_joltik.h
#    aead/joltikeq8048v1/joltikeq8048v1_tweakableBC.h
#    aead/joltikeq8048v1/joltikeq8048v1_encrypt.h
#    aead/joltikeq8048v1/Joltikeq8048v1.h
#    aead/joltikeq9696v1/joltikeq9696v1_api.h
#    aead/joltikeq9696v1/joltikeq9696v1_crypto_aead.h
#    aead/joltikeq9696v1/joltikeq9696v1_joltik.h
#    aead/joltikeq9696v1/joltikeq9696v1_tweakableBC.h
#    aead/joltikeq9696v1/joltikeq9696v1_encrypt.h
#    aead/joltikeq9696v1/Joltikeq9696v1.h
#    aead/joltikneq12864v1/joltikneq12864v1_api.h
#    aead/joltikneq12864v1/joltikneq12864v1_crypto_aead.h
#    aead/joltikneq12864v1/joltikneq12864v1_joltik.h
#    aead/joltikneq12864v1/joltikneq12864v1_tweakableBC.h
#    aead/joltikneq12864v1/joltikneq12864v1_encrypt.h
#    aead/joltikneq12864v1/Joltikneq12864v1.h
#    aead/joltikneq6464v1/joltikneq6464v1_api.h
#    aead/joltikneq6464v1/joltikneq6464v1_crypto_aead.h
#    aead/joltikneq6464v1/joltikneq6464v1_joltik.h
#    aead/joltikneq6464v1/joltikneq6464v1_tweakableBC.h
#    aead/joltikneq6464v1/joltikneq6464v1_encrypt.h
#    aead/joltikneq6464v1/Joltikneq6464v1.h
#    aead/joltikneq8048v1/joltikneq8048v1_api.h
#    aead/joltikneq8048v1/joltikneq8048v1_crypto_aead.h
#    aead/joltikneq8048v1/joltikneq8048v1_joltik.h
#    aead/joltikneq8048v1/joltikneq8048v1_tweakableBC.h
#    aead/joltikneq8048v1/joltikneq8048v1_encrypt.h
#    aead/joltikneq8048v1/Joltikneq8048v1.h
#    aead/joltikneq9696v1/joltikneq9696v1_api.h
#    aead/joltikneq9696v1/joltikneq9696v1_crypto_aead.h
#    aead/joltikneq9696v1/joltikneq9696v1_joltik.h
#    aead/joltikneq9696v1/joltikneq9696v1_tweakableBC.h
#    aead/joltikneq9696v1/joltikneq9696v1_encrypt.h
#    aead/joltikneq9696v1/Joltikneq9696v1.h
#    aead/juliusv1draft/juliusv1draft_api.h
#    aead/juliusv1draft/juliusv1draft_encrypt.h
#    aead/juliusv1draft/Juliusv1draft.h
#    aead/ketjejrv1/ketjejrv1_api.h
#    aead/ketjejrv1/ketjejrv1_brg_endian.h
#    aead/ketjejrv1/ketjejrv1_displayIntermediateValues.h
#    aead/ketjejrv1/ketjejrv1_KeccakF-200-interface.h
#    aead/ketjejrv1/ketjejrv1_KeccakF-200-reference.h
#    aead/ketjejrv1/ketjejrv1_KeccakF-interface.h
#    aead/ketjejrv1/ketjejrv1_KeccakP-200-interface.h
#    aead/ketjejrv1/ketjejrv1_KeccakP-200-reference.h
#    aead/ketjejrv1/ketjejrv1_KeccakP-interface.h
#    aead/ketjejrv1/ketjejrv1_Ket.h
#    aead/ketjejrv1/ketjejrv1_Ketje.h
#    aead/ketjejrv1/ketjejrv1_encrypt.h
#    aead/ketjejrv1/Ketjejrv1.h
#    aead/ketjesrv1/ketjesrv1_api.h
#    aead/ketjesrv1/ketjesrv1_brg_endian.h
#    aead/ketjesrv1/ketjesrv1_displayIntermediateValues.h
#    aead/ketjesrv1/ketjesrv1_KeccakF-400-interface.h
#    aead/ketjesrv1/ketjesrv1_KeccakF-400-reference.h
#    aead/ketjesrv1/ketjesrv1_KeccakF-interface.h
#    aead/ketjesrv1/ketjesrv1_KeccakP-400-interface.h
#    aead/ketjesrv1/ketjesrv1_KeccakP-400-reference.h
#    aead/ketjesrv1/ketjesrv1_KeccakP-interface.h
#    aead/ketjesrv1/ketjesrv1_Ket.h
#    aead/ketjesrv1/ketjesrv1_Ketje.h
#    aead/ketjesrv1/ketjesrv1_encrypt.h
#    aead/ketjesrv1/Ketjesrv1.h
#    aead/kiasueq128v1/kiasueq128v1_api.h
#    aead/kiasueq128v1/kiasueq128v1_crypto_aead.h
#    aead/kiasueq128v1/kiasueq128v1_kiasu.h
#    aead/kiasueq128v1/kiasueq128v1_tweakable_aes.h
#    aead/kiasueq128v1/kiasueq128v1_encrypt.h
#    aead/kiasueq128v1/Kiasueq128v1.h
#    aead/kiasuneq128v1/kiasuneq128v1_api.h
#    aead/kiasuneq128v1/kiasuneq128v1_kiasu.h
#    aead/kiasuneq128v1/kiasuneq128v1_tweakable_aes.h
#    aead/kiasuneq128v1/kiasuneq128v1_encrypt.h
#    aead/kiasuneq128v1/Kiasuneq128v1.h
#    aead/lacv1/lacv1_api.h
#    aead/lacv1/lacv1_encrypt.h
#    aead/lacv1/Lacv1.h
#    aead/lakekeyakv1/lakekeyakv1_api.h
#    aead/lakekeyakv1/lakekeyakv1_brg_endian.h
#    aead/lakekeyakv1/lakekeyakv1_displayIntermediateValues.h
#    aead/lakekeyakv1/lakekeyakv1_KeccakDuplex.h
#    aead/lakekeyakv1/lakekeyakv1_KeccakF-1600-interface.h
#    aead/lakekeyakv1/lakekeyakv1_KeccakF-1600-reference.h
#    aead/lakekeyakv1/lakekeyakv1_KeccakF-interface.h
#    aead/lakekeyakv1/lakekeyakv1_KeccakP-1600-12-interface.h
#    aead/lakekeyakv1/lakekeyakv1_KeccakP-1600-12-reference.h
#    aead/lakekeyakv1/lakekeyakv1_Keyak.h
#    aead/lakekeyakv1/lakekeyakv1_NumberOfParallelInstances.h
#    aead/lakekeyakv1/lakekeyakv1_encrypt.h
#    aead/lakekeyakv1/Lakekeyakv1.h
#    aead/led80n6silcv1/led80n6silcv1_api.h
#    aead/led80n6silcv1/led80n6silcv1_led.h
#    aead/led80n6silcv1/led80n6silcv1_silc.h
#    aead/led80n6silcv1/led80n6silcv1_encrypt.h
#    aead/led80n6silcv1/Led80n6silcv1.h
#    aead/minalpherv1/minalpherv1_api.h
#    aead/minalpherv1/minalpherv1_encrypt.h
#    aead/minalpherv1/Minalpherv1.h
#    aead/morus1280128v1/morus1280128v1_api.h
#    aead/morus1280128v1/morus1280128v1_encrypt.h
#    aead/morus1280128v1/Morus1280128v1.h
#    aead/morus1280256v1/morus1280256v1_api.h
#    aead/morus1280256v1/morus1280256v1_encrypt.h
#    aead/morus1280256v1/Morus1280256v1.h
#    aead/morus640128v1/morus640128v1_api.h
#    aead/morus640128v1/morus640128v1_encrypt.h
#    aead/morus640128v1/Morus640128v1.h
#    aead/norx3241v1/norx3241v1_api.h
#    aead/norx3241v1/norx3241v1_norx_config.h
#    aead/norx3241v1/norx3241v1_norx.h
#    aead/norx3241v1/norx3241v1_norx_util.h
#    aead/norx3241v1/norx3241v1_encrypt.h
#    aead/norx3241v1/Norx3241v1.h
#    aead/norx3261v1/norx3261v1_api.h
#    aead/norx3261v1/norx3261v1_norx_config.h
#    aead/norx3261v1/norx3261v1_norx.h
#    aead/norx3261v1/norx3261v1_norx_util.h
#    aead/norx3261v1/norx3261v1_encrypt.h
#    aead/norx3261v1/Norx3261v1.h
#    aead/norx6441v1/norx6441v1_api.h
#    aead/norx6441v1/norx6441v1_norx_config.h
#    aead/norx6441v1/norx6441v1_norx.h
#    aead/norx6441v1/norx6441v1_norx_util.h
#    aead/norx6441v1/norx6441v1_encrypt.h
#    aead/norx6441v1/Norx6441v1.h
#    aead/norx6444v1/norx6444v1_api.h
#    aead/norx6444v1/norx6444v1_norx_config.h
#    aead/norx6444v1/norx6444v1_norx.h
#    aead/norx6444v1/norx6444v1_norx_util.h
#    aead/norx6444v1/norx6444v1_encrypt.h
#    aead/norx6444v1/Norx6444v1.h
#    aead/norx6461v1/norx6461v1_api.h
#    aead/norx6461v1/norx6461v1_norx_config.h
#    aead/norx6461v1/norx6461v1_norx.h
#    aead/norx6461v1/norx6461v1_norx_util.h
#    aead/norx6461v1/norx6461v1_encrypt.h
#    aead/norx6461v1/Norx6461v1.h
#    aead/oceankeyakv1/oceankeyakv1_api.h
#    aead/oceankeyakv1/oceankeyakv1_brg_endian.h
#    aead/oceankeyakv1/oceankeyakv1_displayIntermediateValues.h
#    aead/oceankeyakv1/oceankeyakv1_KeccakDuplex.h
#    aead/oceankeyakv1/oceankeyakv1_KeccakF-1600-interface.h
#    aead/oceankeyakv1/oceankeyakv1_KeccakF-1600-reference.h
#    aead/oceankeyakv1/oceankeyakv1_KeccakF-interface.h
#    aead/oceankeyakv1/oceankeyakv1_KeccakP-1600-12-interface.h
#    aead/oceankeyakv1/oceankeyakv1_KeccakP-1600-12-reference.h
#    aead/oceankeyakv1/oceankeyakv1_KeccakParallelDuplex.h
#    aead/oceankeyakv1/oceankeyakv1_Keyak.h
#    aead/oceankeyakv1/oceankeyakv1_NumberOfParallelInstances.h
#    aead/oceankeyakv1/oceankeyakv1_ParallelKeccakFs.h
#    aead/oceankeyakv1/oceankeyakv1_testParallelPaSM.h
#    aead/oceankeyakv1/oceankeyakv1_encrypt.h
#    aead/oceankeyakv1/Oceankeyakv1.h
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_api.h
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_inttypes_mingw.h
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_inttypes_win.h
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_md32_common.h
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_omd_api.h
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_omdsha256.h
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_sha256.h
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_stdint_win.h
#    aead/omdsha256k128n96tau128v1/omdsha256k128n96tau128v1_encrypt.h
#    aead/omdsha256k128n96tau128v1/Omdsha256k128n96tau128v1.h
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_api.h
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_inttypes_mingw.h
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_inttypes_win.h
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_md32_common.h
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_omd_api.h
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_omdsha256.h
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_sha256.h
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_stdint_win.h
#    aead/omdsha256k128n96tau64v1/omdsha256k128n96tau64v1_encrypt.h
#    aead/omdsha256k128n96tau64v1/Omdsha256k128n96tau64v1.h
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_api.h
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_inttypes_mingw.h
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_inttypes_win.h
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_md32_common.h
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_omd_api.h
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_omdsha256.h
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_sha256.h
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_stdint_win.h
#    aead/omdsha256k128n96tau96v1/omdsha256k128n96tau96v1_encrypt.h
#    aead/omdsha256k128n96tau96v1/Omdsha256k128n96tau96v1.h
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_api.h
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_inttypes_mingw.h
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_inttypes_win.h
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_md32_common.h
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_omd_api.h
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_omdsha256.h
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_sha256.h
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_stdint_win.h
#    aead/omdsha256k192n104tau128v1/omdsha256k192n104tau128v1_encrypt.h
#    aead/omdsha256k192n104tau128v1/Omdsha256k192n104tau128v1.h
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_api.h
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_inttypes_mingw.h
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_inttypes_win.h
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_md32_common.h
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_omd_api.h
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_omdsha256.h
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_sha256.h
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_stdint_win.h
#    aead/omdsha256k256n104tau160v1/omdsha256k256n104tau160v1_encrypt.h
#    aead/omdsha256k256n104tau160v1/Omdsha256k256n104tau160v1.h
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_api.h
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_inttypes_mingw.h
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_inttypes_win.h
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_md32_common.h
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_omd_api.h
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_omdsha256.h
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_sha256.h
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_stdint_win.h
#    aead/omdsha256k256n248tau256v1/omdsha256k256n248tau256v1_encrypt.h
#    aead/omdsha256k256n248tau256v1/Omdsha256k256n248tau256v1.h
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_api.h
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_inttypes_mingw.h
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_inttypes_win.h
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_omd_api.h
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_omdsha512.h
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_sha512.h
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_stdint_win.h
#    aead/omdsha512k128n128tau128v1/omdsha512k128n128tau128v1_encrypt.h
#    aead/omdsha512k128n128tau128v1/Omdsha512k128n128tau128v1.h
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_api.h
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_inttypes_mingw.h
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_inttypes_win.h
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_omd_api.h
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_omdsha512.h
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_sha512.h
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_stdint_win.h
#    aead/omdsha512k256n256tau256v1/omdsha512k256n256tau256v1_encrypt.h
#    aead/omdsha512k256n256tau256v1/Omdsha512k256n256tau256v1.h
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_api.h
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_inttypes_mingw.h
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_inttypes_win.h
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_omd_api.h
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_omdsha512.h
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_sha512.h
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_stdint_win.h
#    aead/omdsha512k512n256tau256v1/omdsha512k512n256tau256v1_encrypt.h
#    aead/omdsha512k512n256tau256v1/Omdsha512k512n256tau256v1.h
#    aead/paeq128/paeq128_api.h
#    aead/paeq128/paeq128_encrypt.h
#    aead/paeq128/Paeq128.h
#    aead/paeq128t/paeq128t_api.h
#    aead/paeq128t/paeq128t_encrypt.h
#    aead/paeq128t/Paeq128t.h
#    aead/paeq128tnm/paeq128tnm_api.h
#    aead/paeq128tnm/paeq128tnm_encrypt.h
#    aead/paeq128tnm/Paeq128tnm.h
#    aead/paeq160/paeq160_api.h
#    aead/paeq160/paeq160_encrypt.h
#    aead/paeq160/Paeq160.h
#    aead/paeq64/paeq64_api.h
#    aead/paeq64/paeq64_encrypt.h
#    aead/paeq64/Paeq64.h
#    aead/paeq80/paeq80_api.h
#    aead/paeq80/paeq80_encrypt.h
#    aead/paeq80/Paeq80.h
#    aead/pi16cipher096v1/pi16cipher096v1_api.h
#    aead/pi16cipher096v1/pi16cipher096v1_encrypt.h
#    aead/pi16cipher096v1/Pi16cipher096v1.h
#    aead/pi16cipher128v1/pi16cipher128v1_api.h
#    aead/pi16cipher128v1/pi16cipher128v1_encrypt.h
#    aead/pi16cipher128v1/Pi16cipher128v1.h
#    aead/pi32cipher128v1/pi32cipher128v1_api.h
#    aead/pi32cipher128v1/pi32cipher128v1_encrypt.h
#    aead/pi32cipher128v1/Pi32cipher128v1.h
#    aead/pi32cipher256v1/pi32cipher256v1_api.h
#    aead/pi32cipher256v1/pi32cipher256v1_encrypt.h
#    aead/pi32cipher256v1/Pi32cipher256v1.h
#    aead/pi64cipher128v1/pi64cipher128v1_api.h
#    aead/pi64cipher128v1/pi64cipher128v1_encrypt.h
#    aead/pi64cipher128v1/Pi64cipher128v1.h
#    aead/pi64cipher256v1/pi64cipher256v1_api.h
#    aead/pi64cipher256v1/pi64cipher256v1_encrypt.h
#    aead/pi64cipher256v1/Pi64cipher256v1.h
#    aead/pi64cipher256v1oneround/pi64cipher256v1oneround_api.h
#    aead/pi64cipher256v1oneround/pi64cipher256v1oneround_encrypt.h
#    aead/pi64cipher256v1oneround/Pi64cipher256v1oneround.h
#    aead/pi64cipher256v1tworounds/pi64cipher256v1tworounds_api.h
#    aead/pi64cipher256v1tworounds/pi64cipher256v1tworounds_encrypt.h
#    aead/pi64cipher256v1tworounds/Pi64cipher256v1tworounds.h
#    aead/polawisv1/polawisv1_encrypt.h
#    aead/polawisv1/Polawisv1.h
#    aead/ppaev11/ppaev11_api.h
#    aead/ppaev11/ppaev11_encrypt.h
#    aead/ppaev11/Ppaev11.h
#    aead/present80n6silcv1/present80n6silcv1_api.h
#    aead/present80n6silcv1/present80n6silcv1_present.h
#    aead/present80n6silcv1/present80n6silcv1_silc.h
#    aead/present80n6silcv1/present80n6silcv1_encrypt.h
#    aead/present80n6silcv1/Present80n6silcv1.h
#    aead/primatesv1ape120/primatesv1ape120_api.h
#    aead/primatesv1ape120/primatesv1ape120_parameters.h
#    aead/primatesv1ape120/primatesv1ape120_primate.h
#    aead/primatesv1ape120/primatesv1ape120_encrypt.h
#    aead/primatesv1ape120/Primatesv1ape120.h
#    aead/primatesv1ape80/primatesv1ape80_api.h
#    aead/primatesv1ape80/primatesv1ape80_parameters.h
#    aead/primatesv1ape80/primatesv1ape80_primate.h
#    aead/primatesv1ape80/primatesv1ape80_encrypt.h
#    aead/primatesv1ape80/Primatesv1ape80.h
#    aead/primatesv1gibbon120/primatesv1gibbon120_api.h
#    aead/primatesv1gibbon120/primatesv1gibbon120_parameters.h
#    aead/primatesv1gibbon120/primatesv1gibbon120_primate.h
#    aead/primatesv1gibbon120/primatesv1gibbon120_encrypt.h
#    aead/primatesv1gibbon120/Primatesv1gibbon120.h
#    aead/primatesv1gibbon80/primatesv1gibbon80_api.h
#    aead/primatesv1gibbon80/primatesv1gibbon80_parameters.h
#    aead/primatesv1gibbon80/primatesv1gibbon80_primate.h
#    aead/primatesv1gibbon80/primatesv1gibbon80_encrypt.h
#    aead/primatesv1gibbon80/Primatesv1gibbon80.h
#    aead/primatesv1hanuman120/primatesv1hanuman120_api.h
#    aead/primatesv1hanuman120/primatesv1hanuman120_parameters.h
#    aead/primatesv1hanuman120/primatesv1hanuman120_primate.h
#    aead/primatesv1hanuman120/primatesv1hanuman120_encrypt.h
#    aead/primatesv1hanuman120/Primatesv1hanuman120.h
#    aead/primatesv1hanuman80/primatesv1hanuman80_api.h
#    aead/primatesv1hanuman80/primatesv1hanuman80_parameters.h
#    aead/primatesv1hanuman80/primatesv1hanuman80_primate.h
#    aead/primatesv1hanuman80/primatesv1hanuman80_encrypt.h
#    aead/primatesv1hanuman80/Primatesv1hanuman80.h
#    aead/proest128apev1/proest128apev1_ape.h
#    aead/proest128apev1/proest128apev1_api.h
#    aead/proest128apev1/proest128apev1_proest128.h
#    aead/proest128apev1/proest128apev1_zerobytes.h
#    aead/proest128apev1/proest128apev1_encrypt.h
#    aead/proest128apev1/Proest128apev1.h
#    aead/proest128copav1/proest128copav1_api.h
#    aead/proest128copav1/proest128copav1_copa.h
#    aead/proest128copav1/proest128copav1_proest128.h
#    aead/proest128copav1/proest128copav1_zerobytes.h
#    aead/proest128copav1/proest128copav1_encrypt.h
#    aead/proest128copav1/Proest128copav1.h
#    aead/proest128otrv1/proest128otrv1_api.h
#    aead/proest128otrv1/proest128otrv1_otr.h
#    aead/proest128otrv1/proest128otrv1_proest128.h
#    aead/proest128otrv1/proest128otrv1_zerobytes.h
#    aead/proest128otrv1/proest128otrv1_encrypt.h
#    aead/proest128otrv1/Proest128otrv1.h
#    aead/proest256apev1/proest256apev1_ape.h
#    aead/proest256apev1/proest256apev1_api.h
#    aead/proest256apev1/proest256apev1_proest256.h
#    aead/proest256apev1/proest256apev1_zerobytes.h
#    aead/proest256apev1/proest256apev1_encrypt.h
#    aead/proest256apev1/Proest256apev1.h
#    aead/proest256copav1/proest256copav1_api.h
#    aead/proest256copav1/proest256copav1_copa.h
#    aead/proest256copav1/proest256copav1_proest256.h
#    aead/proest256copav1/proest256copav1_zerobytes.h
#    aead/proest256copav1/proest256copav1_encrypt.h
#    aead/proest256copav1/Proest256copav1.h
#    aead/proest256otrv1/proest256otrv1_api.h
#    aead/proest256otrv1/proest256otrv1_otr.h
#    aead/proest256otrv1/proest256otrv1_proest256.h
#    aead/proest256otrv1/proest256otrv1_zerobytes.h
#    aead/proest256otrv1/proest256otrv1_encrypt.h
#    aead/proest256otrv1/Proest256otrv1.h
#    aead/raviyoylav1/raviyoylav1_api.h
#    aead/raviyoylav1/raviyoylav1_encrypt.h
#    aead/raviyoylav1/Raviyoylav1.h
#    aead/riverkeyakv1/riverkeyakv1_api.h
#    aead/riverkeyakv1/riverkeyakv1_brg_endian.h
#    aead/riverkeyakv1/riverkeyakv1_displayIntermediateValues.h
#    aead/riverkeyakv1/riverkeyakv1_KeccakDuplex.h
#    aead/riverkeyakv1/riverkeyakv1_KeccakF-800-interface.h
#    aead/riverkeyakv1/riverkeyakv1_KeccakF-800-reference.h
#    aead/riverkeyakv1/riverkeyakv1_KeccakF-interface.h
#    aead/riverkeyakv1/riverkeyakv1_KeccakP-800-12-interface.h
#    aead/riverkeyakv1/riverkeyakv1_KeccakP-800-12-reference.h
#    aead/riverkeyakv1/riverkeyakv1_Keyak.h
#    aead/riverkeyakv1/riverkeyakv1_NumberOfParallelInstances.h
#    aead/riverkeyakv1/riverkeyakv1_encrypt.h
#    aead/riverkeyakv1/Riverkeyakv1.h
#    aead/sablierv1/sablierv1_api.h
#    aead/sablierv1/sablierv1_authenticate_1.h
#    aead/sablierv1/sablierv1_cipher_1.h
#    aead/sablierv1/sablierv1_encrypt.h
#    aead/sablierv1/Sablierv1.h
#    aead/scream10v1/scream10v1_api.h
#    aead/scream10v1/scream10v1_lbox.h
#    aead/scream10v1/scream10v1_params.h
#    aead/scream10v1/scream10v1_sbox.h
#    aead/scream10v1/scream10v1_encrypt.h
#    aead/scream10v1/Scream10v1.h
#    aead/scream10v2/scream10v2_api.h
#    aead/scream10v2/scream10v2_lbox.h
#    aead/scream10v2/scream10v2_params.h
#    aead/scream10v2/scream10v2_sbox.h
#    aead/scream10v2/scream10v2_encrypt.h
#    aead/scream10v2/Scream10v2.h
#    aead/scream12v1/scream12v1_api.h
#    aead/scream12v1/scream12v1_lbox.h
#    aead/scream12v1/scream12v1_params.h
#    aead/scream12v1/scream12v1_sbox.h
#    aead/scream12v1/scream12v1_encrypt.h
#    aead/scream12v1/Scream12v1.h
#    aead/scream12v2/scream12v2_api.h
#    aead/scream12v2/scream12v2_lbox.h
#    aead/scream12v2/scream12v2_params.h
#    aead/scream12v2/scream12v2_sbox.h
#    aead/scream12v2/scream12v2_encrypt.h
#    aead/scream12v2/Scream12v2.h
#    aead/seakeyakv1/seakeyakv1_api.h
#    aead/seakeyakv1/seakeyakv1_brg_endian.h
#    aead/seakeyakv1/seakeyakv1_displayIntermediateValues.h
#    aead/seakeyakv1/seakeyakv1_KeccakDuplex.h
#    aead/seakeyakv1/seakeyakv1_KeccakF-1600-interface.h
#    aead/seakeyakv1/seakeyakv1_KeccakF-1600-reference.h
#    aead/seakeyakv1/seakeyakv1_KeccakF-interface.h
#    aead/seakeyakv1/seakeyakv1_KeccakP-1600-12-interface.h
#    aead/seakeyakv1/seakeyakv1_KeccakP-1600-12-reference.h
#    aead/seakeyakv1/seakeyakv1_KeccakParallelDuplex.h
#    aead/seakeyakv1/seakeyakv1_Keyak.h
#    aead/seakeyakv1/seakeyakv1_NumberOfParallelInstances.h
#    aead/seakeyakv1/seakeyakv1_ParallelKeccakFs.h
#    aead/seakeyakv1/seakeyakv1_testParallelPaSM.h
#    aead/seakeyakv1/seakeyakv1_encrypt.h
#    aead/seakeyakv1/Seakeyakv1.h
#    aead/shellaes128v1d4n64/shellaes128v1d4n64_aes.h
#    aead/shellaes128v1d4n64/shellaes128v1d4n64_api.h
#    aead/shellaes128v1d4n64/shellaes128v1d4n64_shellaes.h
#    aead/shellaes128v1d4n64/shellaes128v1d4n64_encrypt.h
#    aead/shellaes128v1d4n64/Shellaes128v1d4n64.h
#    aead/shellaes128v1d4n80/shellaes128v1d4n80_aes.h
#    aead/shellaes128v1d4n80/shellaes128v1d4n80_api.h
#    aead/shellaes128v1d4n80/shellaes128v1d4n80_shellaes.h
#    aead/shellaes128v1d4n80/shellaes128v1d4n80_encrypt.h
#    aead/shellaes128v1d4n80/Shellaes128v1d4n80.h
#    aead/shellaes128v1d5n64/shellaes128v1d5n64_aes.h
#    aead/shellaes128v1d5n64/shellaes128v1d5n64_api.h
#    aead/shellaes128v1d5n64/shellaes128v1d5n64_shellaes.h
#    aead/shellaes128v1d5n64/shellaes128v1d5n64_encrypt.h
#    aead/shellaes128v1d5n64/Shellaes128v1d5n64.h
#    aead/shellaes128v1d5n80/shellaes128v1d5n80_aes.h
#    aead/shellaes128v1d5n80/shellaes128v1d5n80_api.h
#    aead/shellaes128v1d5n80/shellaes128v1d5n80_shellaes.h
#    aead/shellaes128v1d5n80/shellaes128v1d5n80_encrypt.h
#    aead/shellaes128v1d5n80/Shellaes128v1d5n80.h
#    aead/shellaes128v1d6n64/shellaes128v1d6n64_aes.h
#    aead/shellaes128v1d6n64/shellaes128v1d6n64_api.h
#    aead/shellaes128v1d6n64/shellaes128v1d6n64_shellaes.h
#    aead/shellaes128v1d6n64/shellaes128v1d6n64_encrypt.h
#    aead/shellaes128v1d6n64/Shellaes128v1d6n64.h
#    aead/shellaes128v1d6n80/shellaes128v1d6n80_aes.h
#    aead/shellaes128v1d6n80/shellaes128v1d6n80_api.h
#    aead/shellaes128v1d6n80/shellaes128v1d6n80_shellaes.h
#    aead/shellaes128v1d6n80/shellaes128v1d6n80_encrypt.h
#    aead/shellaes128v1d6n80/Shellaes128v1d6n80.h
#    aead/shellaes128v1d7n64/shellaes128v1d7n64_aes.h
#    aead/shellaes128v1d7n64/shellaes128v1d7n64_api.h
#    aead/shellaes128v1d7n64/shellaes128v1d7n64_shellaes.h
#    aead/shellaes128v1d7n64/shellaes128v1d7n64_encrypt.h
#    aead/shellaes128v1d7n64/Shellaes128v1d7n64.h
#    aead/shellaes128v1d7n80/shellaes128v1d7n80_aes.h
#    aead/shellaes128v1d7n80/shellaes128v1d7n80_api.h
#    aead/shellaes128v1d7n80/shellaes128v1d7n80_shellaes.h
#    aead/shellaes128v1d7n80/shellaes128v1d7n80_encrypt.h
#    aead/shellaes128v1d7n80/Shellaes128v1d7n80.h
#    aead/shellaes128v1d8n64/shellaes128v1d8n64_aes.h
#    aead/shellaes128v1d8n64/shellaes128v1d8n64_api.h
#    aead/shellaes128v1d8n64/shellaes128v1d8n64_shellaes.h
#    aead/shellaes128v1d8n64/shellaes128v1d8n64_encrypt.h
#    aead/shellaes128v1d8n64/Shellaes128v1d8n64.h
#    aead/shellaes128v1d8n80/shellaes128v1d8n80_aes.h
#    aead/shellaes128v1d8n80/shellaes128v1d8n80_api.h
#    aead/shellaes128v1d8n80/shellaes128v1d8n80_shellaes.h
#    aead/shellaes128v1d8n80/shellaes128v1d8n80_encrypt.h
#    aead/shellaes128v1d8n80/Shellaes128v1d8n80.h
#    aead/silverv1/silverv1_api.h
#    aead/silverv1/silverv1_rijndaelEndianNeutral.h
#    aead/silverv1/silverv1_encrypt.h
#    aead/silverv1/Silverv1.h
#    aead/stribob192r1/stribob192r1_api.h
#    aead/stribob192r1/stribob192r1_crypto_aead.h
#    aead/stribob192r1/stribob192r1_stribob.h
#    aead/stribob192r1/stribob192r1_encrypt.h
#    aead/stribob192r1/Stribob192r1.h
#    aead/tiaoxinv1/tiaoxinv1_aes_round.h
#    aead/tiaoxinv1/tiaoxinv1_api.h
#    aead/tiaoxinv1/tiaoxinv1_tiaoxin-reference.h
#    aead/tiaoxinv1/tiaoxinv1_encrypt.h
#    aead/tiaoxinv1/Tiaoxinv1.h
#    aead/trivia0v1/trivia0v1_api.h
#    aead/trivia0v1/trivia0v1_encrypt.h
#    aead/trivia0v1/Trivia0v1.h
#    aead/trivia128v1/trivia128v1_api.h
#    aead/trivia128v1/trivia128v1_encrypt.h
#    aead/trivia128v1/Trivia128v1.h
#    aead/twine80n6clocv1/twine80n6clocv1_api.h
#    aead/twine80n6clocv1/twine80n6clocv1_cloc.h
#    aead/twine80n6clocv1/twine80n6clocv1_twine.h
#    aead/twine80n6clocv1/twine80n6clocv1_encrypt.h
#    aead/twine80n6clocv1/Twine80n6clocv1.h
#    aead/wheeshtv1mr3fr1t128/wheeshtv1mr3fr1t128_api.h
#    aead/wheeshtv1mr3fr1t128/wheeshtv1mr3fr1t128_encrypt.h
#    aead/wheeshtv1mr3fr1t128/wheeshtv1mr3fr1t128_encrypt.h
#    aead/wheeshtv1mr3fr1t128/Wheeshtv1mr3fr1t128.h
#    aead/wheeshtv1mr3fr1t256/wheeshtv1mr3fr1t256_api.h
#    aead/wheeshtv1mr3fr1t256/wheeshtv1mr3fr1t256_encrypt.h
#    aead/wheeshtv1mr3fr1t256/wheeshtv1mr3fr1t256_encrypt.h
#    aead/wheeshtv1mr3fr1t256/Wheeshtv1mr3fr1t256.h
#    aead/wheeshtv1mr3fr3t256/wheeshtv1mr3fr3t256_api.h
#    aead/wheeshtv1mr3fr3t256/wheeshtv1mr3fr3t256_encrypt.h
#    aead/wheeshtv1mr3fr3t256/wheeshtv1mr3fr3t256_encrypt.h
#    aead/wheeshtv1mr3fr3t256/Wheeshtv1mr3fr3t256.h
#    aead/wheeshtv1mr5fr7t256/wheeshtv1mr5fr7t256_api.h
#    aead/wheeshtv1mr5fr7t256/wheeshtv1mr5fr7t256_encrypt.h
#    aead/wheeshtv1mr5fr7t256/wheeshtv1mr5fr7t256_encrypt.h
#    aead/wheeshtv1mr5fr7t256/Wheeshtv1mr5fr7t256.h
#    aead/yaes128v2/yaes128v2_aes-128.h
#    aead/yaes128v2/yaes128v2_api.h
#    aead/yaes128v2/yaes128v2_encrypt.h
#    aead/yaes128v2/yaes128v2_encrypt.h
#    aead/yaes128v2/Yaes128v2.h
)

set(OTHER_FILES
    aead/acorn128/About.md
    aead/aeadaes128ocbtaglen128v1/About.md
    aead/aeadaes128ocbtaglen64v1/About.md
    aead/aeadaes128ocbtaglen96v1/About.md
    aead/aeadaes192ocbtaglen128v1/About.md
    aead/aeadaes192ocbtaglen64v1/About.md
    aead/aeadaes192ocbtaglen96v1/About.md
    aead/aeadaes256ocbtaglen128v1/About.md
    aead/aeadaes256ocbtaglen64v1/About.md
    aead/aeadaes256ocbtaglen96v1/About.md
    aead/aegis128/About.md
    aead/aegis128l/About.md
    aead/aegis256/About.md
#    aead/aes128avalanchev1/About.md \ # compilaation ok, sigsegv while running
#    aead/aes128cpfbv1/About.md
    aead/aes128gcmv1/About.md
    aead/aes128marble4rv1/About.md
    aead/aes128n12clocv1/About.md
#    aead/aes128n12silcv1/About.md
    aead/aes128n8clocv1/About.md
#    aead/aes128n8silcv1/About.md
    aead/aes128otrpv1/About.md
    aead/aes128otrsv1/About.md
    aead/aes128poetv1aes128/About.md
    aead/aes128poetv1aes4/About.md
    aead/aes192avalanchev1/About.md
    aead/aes256avalanchev1/About.md
#    aead/aes256cpfbv1/About.md
    aead/aes256gcmv1/About.md
    aead/aes256otrpv1/About.md
    aead/aes256otrsv1/About.md
    aead/aescopav1/About.md
    aead/aesjambuv1/About.md
    aead/aezv1/About.md
    aead/aezv3/About.md
    aead/artemia128v1/About.md
    aead/artemia256v1/About.md
    aead/ascon128v1/About.md
    aead/ascon96v1/About.md
    aead/calicov8/About.md
    aead/cba1/About.md
    aead/cba10/About.md
    aead/cba2/About.md
    aead/cba3/About.md
    aead/cba4/About.md
    aead/cba5/About.md
    aead/cba6/About.md
    aead/cba7/About.md
    aead/cba8/About.md
    aead/cba9/About.md
    aead/cmcc22v1/About.md
    aead/cmcc24v1/About.md
    aead/cmcc42v1/About.md
    aead/cmcc44v1/About.md
    aead/cmcc84v1/About.md
    aead/deoxyseq128128v1/About.md
    aead/deoxyseq256128v1/About.md
    aead/deoxysneq128128v1/About.md
    aead/deoxysneq256128v1/About.md
    aead/elmd1000v1/About.md
    aead/elmd1001v1/About.md
    aead/elmd101270v1/About.md
    aead/elmd101271v1/About.md
    aead/elmd500v1/About.md
    aead/elmd501v1/About.md
    aead/elmd51270v1/About.md
    aead/elmd51271v1/About.md
    aead/enchilada128v1/About.md
#    aead/enchilada256v1/About.md
    aead/hs1sivhiv1/About.md
    aead/hs1sivlov1/About.md
    aead/hs1sivv1/About.md
    aead/icepole128av1/About.md
    aead/icepole128v1/About.md
    aead/icepole256av1/About.md
    aead/ifeedaes128n104v1/About.md
    aead/ifeedaes128n96v1/About.md
    aead/iscream12v1/About.md
    aead/iscream12v2/About.md
    aead/iscream14v1/About.md
    aead/iscream14v2/About.md
#    aead/joltikeq12864v1/About.md
#    aead/joltikeq6464v1/About.md
#    aead/joltikeq8048v1/About.md
#    aead/joltikeq9696v1/About.md
#    aead/joltikneq12864v1/About.md
#    aead/joltikneq6464v1/About.md
#    aead/joltikneq8048v1/About.md
#    aead/joltikneq9696v1/About.md
#    aead/juliusv1draft/About.md
#    aead/ketjejrv1/About.md
#    aead/ketjesrv1/About.md
#    aead/kiasueq128v1/About.md
#    aead/kiasuneq128v1/About.md
#    aead/lacv1/About.md
#    aead/lakekeyakv1/About.md
#    aead/led80n6silcv1/About.md
#    aead/minalpherv1/About.md
#    aead/morus1280128v1/About.md
#    aead/morus1280256v1/About.md
#    aead/morus640128v1/About.md
#    aead/norx3241v1/About.md
#    aead/norx3261v1/About.md
#    aead/norx6441v1/About.md
#    aead/norx6444v1/About.md
#    aead/norx6461v1/About.md
#    aead/oceankeyakv1/About.md
#    aead/omdsha256k128n96tau128v1/About.md
#    aead/omdsha256k128n96tau64v1/About.md
#    aead/omdsha256k128n96tau96v1/About.md
#    aead/omdsha256k192n104tau128v1/About.md
#    aead/omdsha256k256n104tau160v1/About.md
#    aead/omdsha256k256n248tau256v1/About.md
#    aead/omdsha512k128n128tau128v1/About.md
#    aead/omdsha512k256n256tau256v1/About.md
#    aead/omdsha512k512n256tau256v1/About.md
#    aead/paeq128/About.md
#    aead/paeq128t/About.md
#    aead/paeq128tnm/About.md
#    aead/paeq160/About.md
#    aead/paeq64/About.md
#    aead/paeq80/About.md
#    aead/pi16cipher096v1/About.md
#    aead/pi16cipher128v1/About.md
#    aead/pi32cipher128v1/About.md
#    aead/pi32cipher256v1/About.md
#    aead/pi64cipher128v1/About.md
#    aead/pi64cipher256v1/About.md
#    aead/pi64cipher256v1oneround/About.md
#    aead/pi64cipher256v1tworounds/About.md
#    aead/polawisv1/About.md
#    aead/ppaev11/About.md
#    aead/present80n6silcv1/About.md
#    aead/primatesv1ape120/About.md
#    aead/primatesv1ape80/About.md
#    aead/primatesv1gibbon120/About.md
#    aead/primatesv1gibbon80/About.md
#    aead/primatesv1hanuman120/About.md
#    aead/primatesv1hanuman80/About.md
#    aead/proest128apev1/About.md
#    aead/proest128copav1/About.md
#    aead/proest128otrv1/About.md
#    aead/proest256apev1/About.md
#    aead/proest256copav1/About.md
#    aead/proest256otrv1/About.md
#    aead/raviyoylav1/About.md
#    aead/riverkeyakv1/About.md
#    aead/sablierv1/About.md
#    aead/scream10v1/About.md
#    aead/scream10v2/About.md
#    aead/scream12v1/About.md
#    aead/scream12v2/About.md
#    aead/seakeyakv1/About.md
#    aead/shellaes128v1d4n64/About.md
#    aead/shellaes128v1d4n80/About.md
#    aead/shellaes128v1d5n64/About.md
#    aead/shellaes128v1d5n80/About.md
#    aead/shellaes128v1d6n64/About.md
#    aead/shellaes128v1d6n80/About.md
#    aead/shellaes128v1d7n64/About.md
#    aead/shellaes128v1d7n80/About.md
#    aead/shellaes128v1d8n64/About.md
#    aead/shellaes128v1d8n80/About.md
#    aead/silverv1/About.md
#    aead/stribob192r1/About.md
#    aead/tiaoxinv1/About.md
#    aead/trivia0v1/About.md
#    aead/trivia128v1/About.md
#    aead/twine80n6clocv1/About.md
#    aead/wheeshtv1mr3fr1t128/About.md
#    aead/wheeshtv1mr3fr1t256/About.md
#    aead/wheeshtv1mr3fr3t256/About.md
#    aead/wheeshtv1mr5fr7t256/About.md
#    aead/yaes128v2/About.md
)

# === source groups ===
source_group(aead     REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/aead/.*")

add_eacirc_project(caesar ${SOURCES} ${HEADERS} ${OTHER_FILES})
target_link_libraries(caesar tinyXML)
