package com.fina.b1s.config;

import com.fina.b1s.b1.B1ServiceLayerProperties;
import com.fina.b1s.b1.B1ProxyProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.net.HttpURLConnection;
import java.security.cert.X509Certificate;

@Configuration
@EnableConfigurationProperties({B1ServiceLayerProperties.class, B1ProxyProperties.class})
public class B1HttpConfig {

    @Bean
    public RestTemplate b1RestTemplate(RestTemplateBuilder builder, B1ServiceLayerProperties properties) {
        return builder
                .requestFactory(() -> new SimpleClientHttpRequestFactory() {
                    private final SSLContext sslContext = trustAllSslContext();

                    @Override
                    protected void prepareConnection(HttpURLConnection connection, String httpMethod) throws java.io.IOException {
                        if (connection instanceof HttpsURLConnection https) {
                            https.setSSLSocketFactory(sslContext.getSocketFactory());
                            https.setHostnameVerifier((hostname, session) -> true);
                        }
                        super.prepareConnection(connection, httpMethod);
                    }
                })
                .setConnectTimeout(properties.connectTimeout())
                .setReadTimeout(properties.readTimeout())
                .build();
    }

    private static SSLContext trustAllSslContext() {
        try {
            TrustManager[] trustManagers = new TrustManager[] {
                    new X509TrustManager() {
                        @Override
                        public void checkClientTrusted(X509Certificate[] chain, String authType) {
                        }

                        @Override
                        public void checkServerTrusted(X509Certificate[] chain, String authType) {
                        }

                        @Override
                        public X509Certificate[] getAcceptedIssuers() {
                            return new X509Certificate[0];
                        }
                    }
            };
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(null, trustManagers, new java.security.SecureRandom());
            return context;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to initialize B1 trust-all SSL context", e);
        }
    }
}
