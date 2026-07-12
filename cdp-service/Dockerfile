# ── Build stage ──────────────────────────────────────────────
FROM eclipse-temurin:17-jdk-jammy AS builder

WORKDIR /build

COPY gradlew .
COPY gradle ./gradle
RUN chmod +x gradlew

COPY settings.gradle.kts build.gradle.kts ./
RUN ./gradlew dependencies --no-daemon -q || true

COPY src ./src
RUN ./gradlew bootJar --no-daemon -q

# ── Runtime stage ─────────────────────────────────────────────
FROM eclipse-temurin:17-jre-jammy

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY --from=builder /build/build/libs/cdp-service.jar app.jar

RUN mkdir -p /app/logs && chown -R appuser:appgroup /app

USER appuser

EXPOSE 5706

ENV LOG_DIR=/app/logs

ENTRYPOINT ["java", \
  "-Xms256m", "-Xmx512m", \
  "-Djava.security.egd=file:/dev/./urandom", \
  "-jar", "app.jar"]
