package com.fina.platform.bo;

import com.fina.platform.bo.BusinessObjectDtos.ObjectDefinitionRequest;
import com.fina.platform.bo.BusinessObjectDtos.ObjectDefinitionResponse;
import com.fina.platform.bo.BusinessObjectDtos.QueryRequest;
import com.fina.platform.bo.BusinessObjectDtos.QueryResponse;
import com.fina.platform.bo.BusinessObjectDtos.RecordRequest;
import com.fina.platform.bo.BusinessObjectDtos.RecordResponse;
import com.fina.platform.bo.BusinessObjectDtos.StoreApiKeyRequest;
import com.fina.platform.bo.BusinessObjectDtos.StoreApiKeyResponse;
import com.fina.platform.bo.BusinessObjectDtos.StoreRequest;
import com.fina.platform.bo.BusinessObjectDtos.StoreResponse;
import com.fina.platform.exception.ApiException;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/bo")
@RequiredArgsConstructor
public class BusinessObjectController {
    private final BusinessObjectService service;

    @Value("${file.api-key:}")
    private String apiKey;

    @GetMapping("/stores")
    public List<StoreResponse> listStores(HttpServletRequest request) {
        requireAdmin(request);
        return service.listStores();
    }

    @PostMapping("/stores")
    public StoreResponse createStore(HttpServletRequest servletRequest,
                                     @RequestBody StoreRequest request) {
        requireAdmin(servletRequest);
        return service.createStore(request);
    }

    @GetMapping("/stores/current")
    public StoreResponse currentStore(HttpServletRequest request) {
        return service.currentStore(service.authenticate(request));
    }

    @PostMapping("/stores/{storeKey}/test")
    public Map<String, Object> testStore(HttpServletRequest request,
                                         @PathVariable String storeKey) {
        requireAdmin(request);
        return service.testStore(storeKey);
    }

    @GetMapping("/stores/{storeKey}/keys")
    public List<StoreApiKeyResponse> listStoreKeys(HttpServletRequest request,
                                                   @PathVariable String storeKey) {
        requireAdmin(request);
        return service.listStoreApiKeys(storeKey);
    }

    @PostMapping("/stores/{storeKey}/keys")
    public StoreApiKeyResponse createStoreKey(HttpServletRequest servletRequest,
                                              @PathVariable String storeKey,
                                              @RequestBody StoreApiKeyRequest request) {
        requireAdmin(servletRequest);
        return service.createStoreApiKey(storeKey, request);
    }

    @PutMapping("/stores/{storeKey}/keys/{keyId}")
    public StoreApiKeyResponse updateStoreKey(HttpServletRequest request,
                                              @PathVariable String storeKey,
                                              @PathVariable Long keyId,
                                              @RequestBody StoreApiKeyRequest body) {
        requireAdmin(request);
        return service.updateStoreApiKey(storeKey, keyId, body);
    }

    @DeleteMapping("/stores/{storeKey}/keys/{keyId}")
    public Map<String, Object> deleteStoreKey(HttpServletRequest request,
                                              @PathVariable String storeKey,
                                              @PathVariable Long keyId) {
        requireAdmin(request);
        return service.deleteStoreApiKey(storeKey, keyId);
    }

    @GetMapping("/objects")
    public List<ObjectDefinitionResponse> listObjects(HttpServletRequest request) {
        return service.listObjects(service.authenticate(request));
    }

    @PostMapping("/objects")
    public ObjectDefinitionResponse createObject(HttpServletRequest servletRequest,
                                                 @RequestBody ObjectDefinitionRequest request) {
        return service.createObject(service.authenticate(servletRequest), request);
    }

    @GetMapping("/objects/{objectKey}")
    public ObjectDefinitionResponse getObject(HttpServletRequest request,
                                              @PathVariable String objectKey) {
        return service.getObject(service.authenticate(request), objectKey);
    }

    @PutMapping("/objects/{objectKey}")
    public ObjectDefinitionResponse updateObject(HttpServletRequest servletRequest,
                                                 @PathVariable String objectKey,
                                                 @RequestBody ObjectDefinitionRequest request) {
        return service.updateObject(service.authenticate(servletRequest), objectKey, request);
    }

    @DeleteMapping("/objects/{objectKey}")
    public Map<String, Object> deleteObject(HttpServletRequest request,
                                            @PathVariable String objectKey) {
        return service.deleteObject(service.authenticate(request), objectKey);
    }

    @PostMapping("/objects/{objectKey}/records")
    public RecordResponse createRecord(HttpServletRequest servletRequest,
                                       @PathVariable String objectKey,
                                       @RequestBody RecordRequest request) {
        return service.createRecord(service.authenticate(servletRequest), objectKey, request);
    }

    @GetMapping("/objects/{objectKey}/records/{id}")
    public RecordResponse getRecord(HttpServletRequest request,
                                    @PathVariable String objectKey,
                                    @PathVariable String id) {
        return service.getRecord(service.authenticate(request), objectKey, id);
    }

    @PatchMapping("/objects/{objectKey}/records/{id}")
    public RecordResponse updateRecord(HttpServletRequest servletRequest,
                                       @PathVariable String objectKey,
                                       @PathVariable String id,
                                       @RequestBody RecordRequest request) {
        return service.updateRecord(service.authenticate(servletRequest), objectKey, id, request);
    }

    @DeleteMapping("/objects/{objectKey}/records/{id}")
    public Map<String, Object> deleteRecord(HttpServletRequest request,
                                            @PathVariable String objectKey,
                                            @PathVariable String id) {
        return service.deleteRecord(service.authenticate(request), objectKey, id);
    }

    @PostMapping("/objects/{objectKey}/records/query")
    public QueryResponse queryRecords(HttpServletRequest servletRequest,
                                      @PathVariable String objectKey,
                                      @RequestBody(required = false) QueryRequest request) {
        return service.queryRecords(service.authenticate(servletRequest), objectKey, request);
    }

    private void requireAdmin(HttpServletRequest request) {
        if (apiKey == null || apiKey.isBlank()) {
            return;
        }
        String presented = headerIgnoreCase(request, "X-Api-Key");
        if (!apiKey.equals(presented)) {
            throw new ApiException(401, "API_KEY_INVALID", "X-Api-Key header is missing or invalid");
        }
    }

    private String headerIgnoreCase(HttpServletRequest request, String name) {
        String exact = request.getHeader(name);
        if (exact != null) {
            return exact;
        }
        java.util.Enumeration<String> names = request.getHeaderNames();
        while (names.hasMoreElements()) {
            String candidate = names.nextElement();
            if (name.equalsIgnoreCase(candidate)) {
                return request.getHeader(candidate);
            }
        }
        return null;
    }
}
