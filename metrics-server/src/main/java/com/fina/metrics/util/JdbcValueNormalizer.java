package com.fina.metrics.util;

import java.lang.reflect.Array;
import java.sql.Blob;
import java.sql.Clob;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

public final class JdbcValueNormalizer {

    private JdbcValueNormalizer() {
    }

    public static Object normalize(Object value) throws SQLException {
        if (value == null) {
            return null;
        }
        if (value instanceof java.sql.Array sqlArray) {
            return normalize(sqlArray.getArray());
        }
        if (value instanceof Clob clob) {
            long length = clob.length();
            if (length > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("CLOB value is too large to return");
            }
            return clob.getSubString(1, (int) length);
        }
        if (value instanceof Blob blob) {
            long length = blob.length();
            if (length > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("BLOB value is too large to return");
            }
            return Base64.getEncoder().encodeToString(blob.getBytes(1, (int) length));
        }
        if (value.getClass().isArray()) {
            int length = Array.getLength(value);
            List<Object> values = new ArrayList<>(length);
            for (int i = 0; i < length; i++) {
                values.add(normalize(Array.get(value, i)));
            }
            return values;
        }
        return value;
    }
}
