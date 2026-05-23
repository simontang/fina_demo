# ── Build stage ──────────────────────────────────────────────
FROM eclipse-temurin:17-jdk-jammy AS builder

WORKDIR /build

# Copy wrapper first for layer-cache friendly dependency download
COPY gradlew .
COPY gradle ./gradle
RUN chmod +x gradlew

COPY settings.gradle.kts build.gradle.kts ./
# Download dependencies without compiling sources
RUN ./gradlew dependencies --no-daemon -q || true

COPY src ./src
RUN ./gradlew bootJar --no-daemon -q

# ── Runtime stage ─────────────────────────────────────────────
FROM eclipse-temurin:17-jre-jammy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY --from=builder /build/build/libs/b1s.jar app.jar

# Create log directory and grant ownership before dropping to appuser
RUN mkdir -p /app/logs && chown -R appuser:appgroup /app

USER appuser

EXPOSE 5705

ENV LOG_DIR=/app/logs

ENTRYPOINT ["java", \
  "-Xms256m", "-Xmx512m", \
  "-Djava.security.egd=file:/dev/./urandom", \
  "-jar", "app.jar"]
