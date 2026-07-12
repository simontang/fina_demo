package com.fina.cdp.dto;

import lombok.Data;

import java.util.Map;

@Data
public class SegmentProcessRequest {

    private Map<String, Object> params;
}
