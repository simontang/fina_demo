package com.fina.platform.dto;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/** Search hit page. Recursive across the tenant's tree (unlike the
 *  one-level directory listing), keyset-paginated newest-first. */
@Data
@Builder
public class FileSearchResult {

    private String query;
    private String prefix;
    private List<FileReceipt> files;
    private boolean truncated;
    private String nextCursor;
    private Integer limit;
}
