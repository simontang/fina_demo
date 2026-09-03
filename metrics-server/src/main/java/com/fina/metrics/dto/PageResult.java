package com.fina.metrics.dto;

import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class PageResult<T> {
    private List<T> items;
    private long total;
    private int page;
    private int pageSize;
}
