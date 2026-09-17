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

val awssdkVersion = "2.25.60"

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-jdbc")
    implementation("org.springframework.boot:spring-boot-starter-validation")
    implementation("org.springframework.boot:spring-boot-starter-actuator")

    implementation("com.baomidou:mybatis-plus-spring-boot3-starter:3.5.5")
    implementation("org.jooq:jooq")

    // Auto DB migration on startup (V* files in classpath:db/migration)
    implementation("org.flywaydb:flyway-core")

    // Metadata storage: PostgreSQL
    runtimeOnly("org.postgresql:postgresql")

    // Object storage (MinIO / S3-compatible) via AWS SDK v2, path-style access
    implementation(platform("software.amazon.awssdk:bom:$awssdkVersion"))
    implementation("software.amazon.awssdk:s3")

    compileOnly("org.projectlombok:lombok")
    annotationProcessor("org.projectlombok:lombok")

    testImplementation("org.springframework.boot:spring-boot-starter-test")
}

tasks.withType<Test> {
    useJUnitPlatform()
}

tasks.bootJar {
    archiveFileName = "platform-service.jar"
}
