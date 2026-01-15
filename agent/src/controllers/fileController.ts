import { FastifyRequest, FastifyReply } from "fastify";
import { MultipartFile } from "@fastify/multipart";
import * as path from "path";
import * as fs from "fs";
import { pipeline } from "stream/promises";

/**
 * File Controller
 * Handles file upload operations
 */

// Configure upload directory
const UPLOAD_DIR =
  process.env.UPLOAD_DIR || path.join(__dirname, "../../uploads");

// Ensure upload directory exists
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

/**
 * File info interface
 */
interface FileInfo {
  id: string;
  originalName: string;
  size: number;
  mimetype: string;
}

/**
 * File upload response interface
 */
interface FileUploadResponse {
  success: boolean;
  message: string;
  id?: string;
  originalName?: string;
  size?: number;
  mimetype?: string;
}

/**
 * Multiple files upload response interface
 */
interface MultipleFilesUploadResponse {
  success: boolean;
  message: string;
  files?: FileInfo[];
  total?: number;
}

/**
 * Generate unique filename with timestamp
 */
function generateUniqueFilename(originalName: string): string {
  const timestamp = Date.now();
  const ext = path.extname(originalName);
  const baseName = path.basename(originalName, ext);
  // Sanitize filename - remove special characters
  const sanitizedBaseName = baseName.replace(/[^a-zA-Z0-9_-]/g, "_");
  return `${sanitizedBaseName}_${timestamp}${ext}`;
}

/**
 * Handle single file upload
 */
export async function uploadFile(
  request: FastifyRequest,
  reply: FastifyReply
): Promise<FileUploadResponse> {
  try {
    const file = await request.file();

    if (!file) {
      return reply.status(400).send({
        success: false,
        message: "No file provided",
      });
    }

    const uniqueFilename = generateUniqueFilename(file.filename);
    const filePath = path.join(UPLOAD_DIR, uniqueFilename);

    // Save file to disk using stream
    await pipeline(file.file, fs.createWriteStream(filePath));

    // Get file stats for size
    const stats = fs.statSync(filePath);

    return {
      success: true,
      message: "File uploaded successfully",
      id: uniqueFilename,
      originalName: file.filename,
      size: stats.size,
      mimetype: file.mimetype,
    };
  } catch (error) {
    request.log.error(error);
    return reply.status(500).send({
      success: false,
      message: `File upload failed: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    });
  }
}

/**
 * Handle multiple files upload
 */
export async function uploadMultipleFiles(
  request: FastifyRequest,
  reply: FastifyReply
): Promise<MultipleFilesUploadResponse> {
  try {
    const files = request.files();
    const uploadedFiles: FileInfo[] = [];

    for await (const file of files) {
      const uniqueFilename = generateUniqueFilename(file.filename);
      const filePath = path.join(UPLOAD_DIR, uniqueFilename);

      // Save file to disk using stream
      await pipeline(file.file, fs.createWriteStream(filePath));

      // Get file stats for size
      const stats = fs.statSync(filePath);

      uploadedFiles.push({
        id: uniqueFilename,
        originalName: file.filename,
        size: stats.size,
        mimetype: file.mimetype,
      });
    }

    if (uploadedFiles.length === 0) {
      return reply.status(400).send({
        success: false,
        message: "No files provided",
      });
    }

    return {
      success: true,
      message: `${uploadedFiles.length} file(s) uploaded successfully`,
      files: uploadedFiles,
      total: uploadedFiles.length,
    };
  } catch (error) {
    request.log.error(error);
    return reply.status(500).send({
      success: false,
      message: `Files upload failed: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    });
  }
}

/**
 * Get list of uploaded files
 */
export async function getUploadedFiles(
  request: FastifyRequest,
  reply: FastifyReply
): Promise<{
  success: boolean;
  message: string;
  files?: string[];
  total?: number;
}> {
  try {
    const files = fs.readdirSync(UPLOAD_DIR);
    return {
      success: true,
      message: "Successfully retrieved uploaded files",
      files,
      total: files.length,
    };
  } catch (error) {
    request.log.error(error);
    return reply.status(500).send({
      success: false,
      message: `Failed to list files: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    });
  }
}

/**
 * Delete an uploaded file
 */
export async function deleteFile(
  request: FastifyRequest<{ Params: { filename: string } }>,
  reply: FastifyReply
): Promise<{ success: boolean; message: string }> {
  try {
    const { filename } = request.params;
    const filePath = path.join(UPLOAD_DIR, filename);

    // Security check: ensure the file is within upload directory
    const resolvedPath = path.resolve(filePath);
    if (!resolvedPath.startsWith(path.resolve(UPLOAD_DIR))) {
      return reply.status(403).send({
        success: false,
        message: "Access denied",
      });
    }

    if (!fs.existsSync(filePath)) {
      return reply.status(404).send({
        success: false,
        message: "File not found",
      });
    }

    fs.unlinkSync(filePath);

    return {
      success: true,
      message: "File deleted successfully",
    };
  } catch (error) {
    request.log.error(error);
    return reply.status(500).send({
      success: false,
      message: `Failed to delete file: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    });
  }
}
