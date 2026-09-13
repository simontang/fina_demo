package com.fina.platform.dto;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Directory-prefix listing. Directories are aggregated in SQL; files are
 * keyset-paginated (order by filename) so a directory with millions of
 * objects is walked page by page instead of loaded whole.
 */
@Data
@Builder
public class PathListing {

    private String prefix;
    private List<String> directories;
    private List<FileReceipt> files;
    /** true when more files exist after this page — pass nextCursor to continue. */
    private boolean truncated;
    /** opaque cursor (last filename of this page); null when not truncated. */
    private String nextCursor;
    private Integer limit;
}
