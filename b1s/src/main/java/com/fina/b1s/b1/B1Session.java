package com.fina.b1s.b1;

import java.time.Instant;
import java.util.List;

record B1Session(String sessionId, String routeId, Instant createdAt) {

    List<String> cookies() {
        if (routeId == null || routeId.isBlank()) {
            return List.of("B1SESSION=" + sessionId);
        }
        return List.of("B1SESSION=" + sessionId, "ROUTEID=" + routeId);
    }
}
