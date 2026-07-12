package com.fina.cdp;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.fina.cdp.mapper")
public class CdpServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(CdpServiceApplication.class, args);
    }
}
