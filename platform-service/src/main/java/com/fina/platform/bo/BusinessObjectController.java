package com.fina.platform.bo;

import com.fina.platform.bo.BusinessObjectDtos.ObjectDefinitionRequest;
import com.fina.platform.bo.BusinessObjectDtos.ObjectDefinitionResponse;
import com.fina.platform.bo.BusinessObjectDtos.QueryRequest;
import com.fina.platform.bo.BusinessObjectDtos.QueryResponse;
import com.fina.platform.bo.BusinessObjectDtos.RecordRequest;
import com.fina.platform.bo.BusinessObjectDtos.RecordResponse;
import com.fina.platform.bo.BusinessObjectDtos.StoreGrantRequest;
import com.fina.platform.bo.BusinessObjectDtos.StoreGrantResponse;
import com.fina.platform.bo.BusinessObjectDtos.StoreRequest;
import com.fina.platform.bo.BusinessObjectDtos.StoreResponse;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
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

    @GetMapping("/stores")
    public List<StoreResponse> listStores() {
        return service.listStores();
    }

    @PostMapping("/stores")
    public StoreResponse createStore(@RequestBody StoreRequest request) {
        return service.createStore(request);
    }

    @PostMapping("/stores/{storeKey}/test")
    public Map<String, Object> testStore(@PathVariable String storeKey) {
        return service.testStore(storeKey);
    }

    @GetMapping("/stores/{storeKey}/grants")
    public List<StoreGrantResponse> listStoreGrants(@PathVariable String storeKey) {
        return service.listStoreGrants(storeKey);
    }

    @PostMapping("/stores/{storeKey}/grants")
    public StoreGrantResponse grantStore(@PathVariable String storeKey,
                                         @RequestBody StoreGrantRequest request) {
        return service.createOrUpdateStoreGrant(storeKey, request);
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
}
