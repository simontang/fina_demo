package com.fina.platform.dto;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Pseudo-directory listing: files directly under `prefix` plus the next
 * directory segment aggregate. There is no folder entity — directories are
 * derived from path prefixes on the fly.
 */
@Data
@Builder
public class PathListing {

    private String prefix;
    private List<String> directories;
    private List<FileReceipt> files;
}
