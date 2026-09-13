package com.fina.platform.dto;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Result of "find files under a folder": the folder's immediate sub-folders
 * (only when not recursive) plus one page of files, newest first.
 */
@Data
@Builder
public class PathListing {

    /** Folder the query was scoped to ("" = the tenant's root). */
    private String path;
    /** true when descendants were included. */
    private boolean recursive;
    /** name-substring filter that was applied, if any. */
    private String query;
    /** next-level folder names; empty for recursive queries. */
    private List<String> directories;
    private List<FileReceipt> files;
    private Integer limit;
    /** true when more files exist — pass nextCursor back to continue. */
    private boolean truncated;
    /** opaque continuation handle (null when truncated=false). */
    private String nextCursor;
}
