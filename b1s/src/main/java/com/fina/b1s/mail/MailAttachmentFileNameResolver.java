package com.fina.b1s.mail;

import jakarta.mail.internet.ContentType;
import jakarta.mail.internet.MimeUtility;
import org.springframework.util.StringUtils;

import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

final class MailAttachmentFileNameResolver {

    private MailAttachmentFileNameResolver() {
    }

    static String resolve(String partFileName, String contentType, byte[] bytes) {
        String decodedPartFileName = normalizeCandidate(decodeMimeText(partFileName));
        String decodedContentTypeName = normalizeCandidate(decodeMimeText(extractContentTypeName(contentType)));

        String candidate = chooseBestCandidate(decodedPartFileName, decodedContentTypeName);
        String extension = inferExtension(contentType, bytes);
        if (!StringUtils.hasText(candidate)) {
            return "attachment" + (extension != null ? extension : ".bin");
        }
        if (!hasExtension(candidate) && extension != null) {
            return candidate + extension;
        }
        return candidate;
    }

    private static String chooseBestCandidate(String first, String second) {
        if (StringUtils.hasText(first) && hasExtension(first)) {
            return first;
        }
        if (StringUtils.hasText(second) && hasExtension(second)) {
            return second;
        }
        return StringUtils.hasText(first) ? first : second;
    }

    private static String extractContentTypeName(String contentType) {
        if (!StringUtils.hasText(contentType)) {
            return null;
        }
        try {
            String parsed = new ContentType(contentType).getParameter("name");
            if (StringUtils.hasText(parsed)) {
                return parsed;
            }
        } catch (Exception ignored) {
            // Fall back to a tolerant parser below. Mail clients often emit loose
            // headers around encoded filenames.
        }
        return extractLooseParameter(contentType, "name");
    }

    private static String extractLooseParameter(String header, String parameterName) {
        String[] parts = header.split(";");
        for (String part : parts) {
            int equals = part.indexOf('=');
            if (equals < 0) {
                continue;
            }
            String name = part.substring(0, equals).trim();
            String value = stripQuotes(part.substring(equals + 1).trim());
            if (name.equalsIgnoreCase(parameterName)) {
                return value;
            }
            if (name.equalsIgnoreCase(parameterName + "*")) {
                return decodeRfc5987(value);
            }
        }
        return null;
    }

    private static String decodeMimeText(String value) {
        if (!StringUtils.hasText(value)) {
            return null;
        }
        try {
            return MimeUtility.decodeText(value);
        } catch (UnsupportedEncodingException ignored) {
            return value;
        }
    }

    private static String decodeRfc5987(String value) {
        String stripped = stripQuotes(value);
        int firstQuote = stripped.indexOf('\'');
        int secondQuote = firstQuote >= 0 ? stripped.indexOf('\'', firstQuote + 1) : -1;
        if (firstQuote <= 0 || secondQuote <= firstQuote) {
            return stripped;
        }
        String charsetName = stripped.substring(0, firstQuote);
        String encoded = stripped.substring(secondQuote + 1);
        try {
            return URLDecoder.decode(encoded, Charset.forName(charsetName));
        } catch (Exception ignored) {
            return URLDecoder.decode(encoded, StandardCharsets.UTF_8);
        }
    }

    private static String normalizeCandidate(String value) {
        if (!StringUtils.hasText(value)) {
            return null;
        }
        String normalized = stripQuotes(value)
                .replace('\u00a0', ' ')
                .replace('\u202f', ' ')
                .replace('\r', ' ')
                .replace('\n', ' ')
                .replace('\t', ' ')
                .trim()
                .replaceAll(" {2,}", " ");
        int slash = Math.max(normalized.lastIndexOf('/'), normalized.lastIndexOf('\\'));
        if (slash >= 0 && slash + 1 < normalized.length()) {
            normalized = normalized.substring(slash + 1).trim();
        }
        return StringUtils.hasText(normalized) ? normalized : null;
    }

    private static String stripQuotes(String value) {
        if (value == null) {
            return null;
        }
        String stripped = value.trim();
        if (stripped.length() >= 2 && stripped.startsWith("\"") && stripped.endsWith("\"")) {
            return stripped.substring(1, stripped.length() - 1);
        }
        return stripped;
    }

    private static boolean hasExtension(String fileName) {
        if (!StringUtils.hasText(fileName)) {
            return false;
        }
        int lastDot = fileName.lastIndexOf('.');
        return lastDot > 0 && lastDot < fileName.length() - 1;
    }

    private static String inferExtension(String contentType, byte[] bytes) {
        String normalized = normalizeContentType(contentType);
        if ("application/pdf".equals(normalized) || hasPdfMagic(bytes)) {
            return ".pdf";
        }
        if ("image/png".equals(normalized)) {
            return ".png";
        }
        if ("image/jpeg".equals(normalized)) {
            return ".jpg";
        }
        if ("image/webp".equals(normalized)) {
            return ".webp";
        }
        if ("image/tiff".equals(normalized)) {
            return ".tiff";
        }
        if ("application/msword".equals(normalized)) {
            return ".doc";
        }
        if ("application/vnd.openxmlformats-officedocument.wordprocessingml.document".equals(normalized)) {
            return ".docx";
        }
        return null;
    }

    private static String normalizeContentType(String contentType) {
        if (!StringUtils.hasText(contentType)) {
            return "";
        }
        String normalized = contentType.trim();
        int parameterStart = normalized.indexOf(';');
        if (parameterStart >= 0) {
            normalized = normalized.substring(0, parameterStart).trim();
        }
        return normalized.toLowerCase(Locale.ROOT);
    }

    private static boolean hasPdfMagic(byte[] bytes) {
        return bytes != null
                && bytes.length >= 4
                && bytes[0] == '%'
                && bytes[1] == 'P'
                && bytes[2] == 'D'
                && bytes[3] == 'F';
    }
}
