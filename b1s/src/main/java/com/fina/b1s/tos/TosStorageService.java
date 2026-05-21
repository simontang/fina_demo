package com.fina.b1s.tos;

import java.io.InputStream;

public interface TosStorageService {

    UploadResult upload(String key, InputStream content, long contentLength);

    record UploadResult(String bucket, String key, String url) {
    }
}
