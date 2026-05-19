package com.fina.b1s;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.fina.b1s.mapper")
public class B1sApplication {

    public static void main(String[] args) {
        SpringApplication.run(B1sApplication.class, args);
    }
}
