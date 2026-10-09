# geedl

Download Google Earth Engine images and collections locally via Google Drive.

**geedl** solves the problem of downloading large GEE datasets that exceed Google Drive storage limits by automatically managing the download pipeline: submitting export tasks in batches, monitoring their progress, downloading completed files, and deleting them from Drive to make room for more.

## Features

- Download any GEE image or image collection with a single command
- Auto-detects asset type (Image vs ImageCollection)
- Uses native resolution by default (no need to specify scale)
- Automatic tile-based processing for large regions
- Resume support for crash recovery
- Parallel downloads with configurable workers (default: 8)
- ROI filtering with any geospatial file format
- Date, band, and metadata filtering for collections
- Post-completion monitoring to catch straggler files
- **Direct download** for small tiles (bypasses Drive entirely via `getDownloadURL()`)
- **Storage-aware pipeline** that pauses task submission when Drive is running low
- **Adaptive concurrency** that auto-tunes to your GEE quota limits
- **EECU usage tracking** with configurable warnings
- **Google Cloud Storage export** as alternative to Drive (optional)

## Installation

```bash
pip install geedl

# With Google Cloud Storage support (optional)
pip install geedl[gcs]
```

## Quick Start

### Authentication Setup

1. Create a Google Cloud project and enable:
   - Earth Engine API
   - Google Drive API
   - Cloud Resource Manager API

2. Create OAuth2 credentials (Desktop application) and download as JSON

3. Run authentication:
```bash
geedl auth --credentials /path/to/credentials.json --test
```

### Download Examples

```bash
# Download SRTM elevation (auto-detected as Image, uses native 30m resolution)
geedl download USGS/SRTMGL1_003 -o ./output

# Download Sentinel-1 collection with filters
geedl download COPERNICUS/S1_GRD -o ./output \
    --start-date 2023-01-01 --end-date 2023-12-31 \
    --bands VV VH \
    --filter instrumentMode eq IW

# Download with ROI and custom resolution
geedl download LANDSAT/LC08/C02/T1_L2 -o ./output \
    --roi region.shp --scale 30 --resume

# Download PALSAR data for a bounding box
geedl download JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH -o ./output \
    --bounds -70 -10 -60 0 \
    --start-date 2020-01-01

# Direct download (bypass Drive for small tiles)
geedl download USGS/SRTMGL1_003 -o ./output \
    --download-method direct --bounds -70 -10 -69 -9

# Export to Google Cloud Storage instead of Drive
geedl download COPERNICUS/S1_GRD -o ./output \
    --export-target gcs --gcs-bucket my-bucket \
    --start-date 2023-01-01 --end-date 2023-12-31

# Time interval processing (yearly subdirectories)
geedl download COPERNICUS/S1_GRD -o ./output \
    --start-date 2020-01-01 --end-date 2023-12-31 \
    --interval 1 --interval-unit years
```

### Python API

```python
from geedl import GEEDownloader

downloader = GEEDownloader(
    output_dir="./output",
    max_concurrent_tasks=10,
    download_workers=8,
)

# Download any asset (auto-detects type, uses native resolution)
downloader.download(
    asset_id="COPERNICUS/S1_GRD",
    start_date="2023-01-01",
    end_date="2023-12-31",
    roi_file="region.shp",
    bands=["VV", "VH"],
    filters=[
        ("instrumentMode", "eq", "IW"),
    ],
    resume=True,
)
```

## How It Works

1. **Account Diagnostics**: Queries actual Drive storage and GEE task status at startup
2. **Asset Detection**: Automatically determines if the asset is an Image or ImageCollection
3. **Native Resolution**: Reads the asset's native scale (no guessing required)
4. **Tile Generation**: Large regions are divided into manageable tiles
5. **Smart Routing**: Small tiles are downloaded directly via `getDownloadURL()` (bypasses Drive); larger tiles go through the batch export pipeline
6. **Storage-Aware Submission**: Monitors Drive usage and pauses task submission when approaching the headroom threshold
7. **Adaptive Concurrency**: Auto-tunes the number of concurrent GEE tasks based on queue behavior
8. **Task Submission**: Export tasks are submitted to GEE (or GCS) with rate limiting
9. **Status Monitoring**: Monitors task completion with EECU usage tracking
10. **Parallel Downloads**: Completed files are downloaded in parallel, largest first to reclaim Drive space faster
11. **Cleanup**: Files are deleted from Drive/GCS after successful download
12. **Post-completion Wait**: Continues checking for straggler files (default: 30 minutes)

## Download Methods

### Drive (default)
The standard pipeline: GEE exports to Drive, geedl downloads and deletes. Works for any file size but constrained by Drive storage (15 GB free).

### Direct
Uses `ee.Image.getDownloadURL()` to stream tiles directly from GEE to local disk. No Drive storage used. Best for tiles under ~32 MB. Uses interactive EECU quota (separate from batch).

### Auto (recommended)
Estimates tile size and automatically chooses direct download for small tiles, Drive export for large ones. Set with `--download-method auto` (the default).

## Export Targets

### Google Drive (default)
No extra setup required. geedl monitors Drive storage and pauses submissions to avoid exceeding your account limit.

### Google Cloud Storage
For users with a GCP project. Removes the 15 GB Drive constraint. The GCS free tier offers 5 GB storage + 100 GB/month outbound transfer.

```bash
pip install geedl[gcs]
geedl download ASSET_ID -o ./output --export-target gcs --gcs-bucket my-bucket
```

## Performance Tuning

| Parameter | Default | Flag | Notes |
|-----------|---------|------|-------|
| Download workers | 8 | `--workers` | Increase for fast connections |
| Download chunk size | 50 MB | `--chunk-size` | Larger = fewer HTTP round-trips |
| Max concurrent tasks | 10 | `--max-tasks` | Auto-tunes down if GEE queues tasks |
| Submission delay | 2.0s | `--submission-delay` | Between GEE task submissions |
| Check interval | 30s | `--check-interval` | Between status checks |
| File dimensions | 2560x2560 | `--file-dimensions` | Export tile size in pixels |
| Direct threshold | 32 MB | `--direct-threshold` | Max size for direct download |
| Storage headroom | 15% | `--storage-headroom` | Min free Drive space (% of limit) |
| EECU warning | 100h | `--eecu-warn` | Warn at this session EECU usage |
| Post-completion wait | 1800s | `--post-wait` | Wait for straggler files |

## GEE Quota Notes

As of April 2026, noncommercial GEE projects use tiered EECU quotas:

| Tier | Monthly EECU-Hours | Concurrent Batch Tasks |
|------|-------------------|----------------------|
| Community (default) | 150 | ~2 |
| Contributor | 1,000 | ~8 |
| Partner | 100,000 | ~20+ |

geedl's adaptive concurrency automatically adjusts to your account's limits. If tasks stay queued (READY state) for too long, it reduces the concurrency cap. If all tasks are running, it gradually increases it.

## Configuration

Default configuration directory: `~/.geedl/`
- `credentials.json` - OAuth2 credentials from Google Cloud Console
- `token.json` - Cached authentication token (auto-generated)

## License

MIT
