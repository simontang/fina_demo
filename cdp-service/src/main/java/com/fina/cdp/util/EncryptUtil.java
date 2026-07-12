package com.fina.cdp.util;

import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Base64;

public final class EncryptUtil {

    private static final String ALGORITHM = "AES/CBC/PKCS5Padding";
    private static final byte[] FIXED_IV = "fina-metrics-iv!".getBytes(StandardCharsets.UTF_8);

    private EncryptUtil() {
    }

    public static String encrypt(String plainText, String secretKey) {
        try {
            byte[] keyBytes = deriveKey(secretKey);
            SecretKeySpec keySpec = new SecretKeySpec(keyBytes, "AES");
            IvParameterSpec ivSpec = new IvParameterSpec(FIXED_IV);
            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.ENCRYPT_MODE, keySpec, ivSpec);
            byte[] encrypted = cipher.doFinal(plainText.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(encrypted);
        } catch (Exception e) {
            throw new RuntimeException("Encryption failed", e);
        }
    }

    public static String decrypt(String cipherText, String secretKey) {
        try {
            byte[] keyBytes = deriveKey(secretKey);
            SecretKeySpec keySpec = new SecretKeySpec(keyBytes, "AES");
            IvParameterSpec ivSpec = new IvParameterSpec(FIXED_IV);
            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.DECRYPT_MODE, keySpec, ivSpec);
            byte[] decoded = Base64.getDecoder().decode(cipherText);
            return new String(cipher.doFinal(decoded), StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new RuntimeException("Decryption failed", e);
        }
    }

    private static byte[] deriveKey(String key) throws Exception {
        MessageDigest sha = MessageDigest.getInstance("SHA-256");
        byte[] keyBytes = sha.digest(key.getBytes(StandardCharsets.UTF_8));
        return Arrays.copyOf(keyBytes, 32);
    }
}
