plugins {
    java
    id("org.springframework.boot") version "3.2.3"
    id("io.spring.dependency-management") version "1.1.4"
}

group = "com.fina"
version = "1.0.0"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

repositories {
    mavenCentral()
}

val mybatisPlusVersion = "3.5.5"
val mssqlJdbcVersion = "12.6.1.jre11"
val awsSdkVersion = "2.25.70"

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-jdbc")
    implementation("org.springframework.boot:spring-boot-starter-validation")
    implementation("org.springframework.boot:spring-boot-starter-actuator")
    implementation("org.springframework.boot:spring-boot-starter-mail")
    implementation("org.apache.pdfbox:pdfbox:2.0.31")

    implementation("com.baomidou:mybatis-plus-spring-boot3-starter:$mybatisPlusVersion")

    // Master datasource: PostgreSQL
    runtimeOnly("org.postgresql:postgresql")

    // Dynamic datasource: SAP B1 SQL Server
    implementation("com.microsoft.sqlserver:mssql-jdbc:$mssqlJdbcVersion")

    // Volcano Engine TOS S3-compatible API for email attachments
    implementation("software.amazon.awssdk:s3:$awsSdkVersion")

    compileOnly("org.projectlombok:lombok")
    annotationProcessor("org.projectlombok:lombok")

    testImplementation("org.springframework.boot:spring-boot-starter-test")
}

tasks.withType<Test> {
    useJUnitPlatform()
}

tasks.bootJar {
    archiveFileName = "b1s.jar"
}
