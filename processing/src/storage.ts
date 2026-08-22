/**
 * Cloudflare R2 storage integration for uploading processed splats.
 * Uses @aws-sdk/client-s3 with R2's S3-compatible API.
 *
 * Required environment variables:
 *   R2_ACCOUNT_ID  - Cloudflare account ID
 *   R2_ACCESS_KEY  - R2 API token access key
 *   R2_SECRET_KEY  - R2 API token secret key
 *   R2_BUCKET      - R2 bucket name
 *   R2_PUBLIC_URL  - (optional) Custom domain for public URLs
 */

import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import fs from 'fs';
import path from 'path';

function getR2Config() {
  const accountId = process.env.R2_ACCOUNT_ID;
  const accessKey = process.env.R2_ACCESS_KEY;
  const secretKey = process.env.R2_SECRET_KEY;
  const bucket = process.env.R2_BUCKET;

  if (!accountId || !accessKey || !secretKey || !bucket) {
    throw new Error(
      'R2 storage not configured. Set R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET'
    );
  }

  return { accountId, accessKey, secretKey, bucket };
}

function createS3Client(): S3Client {
  const config = getR2Config();
  return new S3Client({
    region: 'auto',
    endpoint: `https://${config.accountId}.r2.cloudflarestorage.com`,
    credentials: {
      accessKeyId: config.accessKey,
      secretAccessKey: config.secretKey,
    },
  });
}

function getPublicUrl(key: string): string {
  const customDomain = process.env.R2_PUBLIC_URL;
  if (customDomain) {
    return `${customDomain}/${key}`;
  }
  const config = getR2Config();
  return `https://${config.bucket}.${config.accountId}.r2.dev/${key}`;
}

/**
 * Upload a PLY file to R2.
 */
export async function uploadSplat(projectId: string, plyFilePath: string): Promise<string> {
  const config = getR2Config();
  const client = createS3Client();

  const key = `splats/${projectId}/model.ply`;
  const fileStream = fs.createReadStream(plyFilePath);
  const fileSize = fs.statSync(plyFilePath).size;

  console.log(`[storage] Uploading PLY to R2: ${key} (${(fileSize / 1024 / 1024).toFixed(1)} MB)`);

  await client.send(new PutObjectCommand({
    Bucket: config.bucket,
    Key: key,
    Body: fileStream,
    ContentType: 'application/octet-stream',
    ContentLength: fileSize,
  }));

  const url = getPublicUrl(key);
  console.log(`[storage] PLY uploaded: ${url}`);
  return url;
}

/**
 * Upload a thumbnail image to R2.
 */
export async function uploadThumbnail(projectId: string, imagePath: string): Promise<string> {
  const config = getR2Config();
  const client = createS3Client();

  const ext = path.extname(imagePath).toLowerCase();
  const contentType = ext === '.png' ? 'image/png' : 'image/jpeg';
  const key = `splats/${projectId}/thumbnail${ext}`;
  const fileStream = fs.createReadStream(imagePath);
  const fileSize = fs.statSync(imagePath).size;

  console.log(`[storage] Uploading thumbnail to R2: ${key}`);

  await client.send(new PutObjectCommand({
    Bucket: config.bucket,
    Key: key,
    Body: fileStream,
    ContentType: contentType,
    ContentLength: fileSize,
  }));

  const url = getPublicUrl(key);
  console.log(`[storage] Thumbnail uploaded: ${url}`);
  return url;
}

/**
 * Get the public URL for a project's splat file.
 */
export function getSplatUrl(projectId: string): string {
  return getPublicUrl(`splats/${projectId}/model.ply`);
}

/**
 * Check if R2 is configured.
 */
export function isR2Configured(): boolean {
  return !!(
    process.env.R2_ACCOUNT_ID &&
    process.env.R2_ACCESS_KEY &&
    process.env.R2_SECRET_KEY &&
    process.env.R2_BUCKET
  );
}
