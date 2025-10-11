# ===============================================
# 🚀 Dropbox Uploader (Resumable + Auto-Retry + Progress)
# ===============================================
import os
import time
import dropbox
from tqdm import tqdm

# --- Configuration ---
DROPBOX_TOKEN = "sl.u.AGBgdN1jzLj5bdARcB_NNzivPtxMfrQzUFefAziVzWtukNvkoGUphQkixL5zFLVqwQnoQ-VQ9aIRLZ0nO1GmZzqpNcn_IG1RBzCXxr87L7hogAYUbwEYSozSRmHyN0eB3QYyBtxbwTdijRXfIy7r6ycCUGsaFZAfRCa1J_Vjf72x6jIu9i_pSkQ4SOweFev5TWV7ECEUJY7FQSeesI0QR-iHu2T4Pqvi_V-ZTyinmXN66h7O8hox4asgFfJ1RQVljNJpdc32La5xLAOyXxZN6EFPy68gVCmFTAn99vrMTICZ32AAYladcUojtDwWAiWT0v0UmkSV_6kwZcBvHMoXJ_gzfqjQgjkT8-uLr-TnlE32QYajcLaRyigVwTIOd3-S0sbyksNH4wP5VY4aGhLj0uKwyZZjECojsKf-kzOBL9cTvRWh4wBQpKvTeJ63WPTOYc-RT78KJzQR2rvZCDq82g5Z6pf0V8rO80XivyvhblBFb92PJgfhzrZTULCY53UUP2UVnAt-lvvMZmF43i8NitxbdIRXvjEogvlDM3GZgsOC0OnsBg3vBm_1N2frop7uT40TZLszF0QqHU8fIu2vnGtSWlcAELtNRDhxJPhCam25AyAtU9umi3zYe72sY_0g3X3Q--ceqZPv4VV_0CgsXo9ppUy06P6Vc7sEm3Jg5Xwf3VM545x8IyqgvDSzy--LzMfzGx7PFCNMe_uceGTZkOGeSpl2jUrajxpbj3NnqZ1aiRly3OVQHtpKDAyY7iGW97sqpGWGEgdMBp4DdLpsJlcxU84SkYvps52oGc0Qd7UMVzFFyIcOYJkd-jMpKMv_ahYmpnkLrfjcguUHdgj-X1G_SwYYGQ8_3Sq1KQHNI3I3t_k6GOQ53ECJTrkGXDqjWUmpqeOF-hoR1Vb3UCp2RJFgOVjVCp-rvWfEztGPh_wPWzW9Li8jr63WztMK9SNZJ4f1LZAhvd_T2Jua7M7OHLc9XzF3lfeF3zBi03Z4UygIIT1RMjaYxWUoTDleTTU4CWWRLX57xvml8ILo-Gdvybnkc8ZVHvannjQrbVlINE-jf8NAW84btFVOHsijO_OxLsIn53SFrRDmkO5_bK6Ir35B1OnburUlEvnZVB1otG0FxWOqtk1i1Yu8oASncqsLwCAAuUkA7LcZoL4z-4Sv04Wvu6QOUu446ZEegiaXmSlDbuXOZj82Fn72s3BA0aeb5rwfB3b4T9-FhJec4LurIwcITxp71ThJguy-XFJ6LHv_Zxltm6h8Bzu8vdi9WcZJp8oQZPYq5GGdUiE6qRL-vZfCe6hcJJfJHfKKOIl_Ktj2Wj75cUp8aFh9izdLWcjqrXtCJCOb6nj1QYDeX2VdBRhQhAr0zGpMvIXlsXDUzaV-AzVdWBJzM7RJfDN3I7GTivOjCSgOvGTgEvwe3ZpkBxOgGAmCfb4C5bTxC3ncdQnExg"  # Replace with your current safe token
LOCAL_MODEL_DIR = "./models/checkpoint-3500"
DROPBOX_MODEL_DIR = "/models/checkpoint-3500"
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB chunks (safe for macOS/LibreSSL)
MAX_RETRIES = 5                 # retry up to 5 times on network errors
RETRY_DELAY = 10                # wait 10 s before retrying

# --- Connect ---
dbx = dropbox.Dropbox(DROPBOX_TOKEN)
acc = dbx.users_get_current_account()
print(f"✅ Connected to Dropbox as: {acc.name.display_name}")

# --- Ensure target folder ---
def ensure_folder(path):
    try:
        dbx.files_get_metadata(path)
    except dropbox.exceptions.ApiError:
        dbx.files_create_folder_v2(path)
        print(f"📁 Created folder: {path}")

ensure_folder(DROPBOX_MODEL_DIR)

# --- Resumable upload function ---
def upload_large_file(local_path, dropbox_path):
    file_size = os.path.getsize(local_path)
    uploaded_bytes = 0

    for attempt in range(MAX_RETRIES):
        try:
            with open(local_path, "rb") as f:
                # Skip bytes if file partially uploaded
                if uploaded_bytes:
                    f.seek(uploaded_bytes)

                if file_size <= CHUNK_SIZE:
                    dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))
                    print(f"✅ Uploaded {os.path.basename(local_path)} (single chunk)")
                    return

                # Start session
                session_start = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
                cursor = dropbox.files.UploadSessionCursor(session_id=session_start.session_id, offset=f.tell())
                commit = dropbox.files.CommitInfo(path=dropbox_path, mode=dropbox.files.WriteMode("overwrite"))

                with tqdm(total=file_size, unit="B", unit_scale=True, desc=os.path.basename(local_path)) as pbar:
                    pbar.update(f.tell())
                    while f.tell() < file_size:
                        chunk = f.read(CHUNK_SIZE)
                        if not chunk:
                            break

                        for retry in range(MAX_RETRIES):
                            try:
                                if (file_size - f.tell()) <= CHUNK_SIZE:
                                    dbx.files_upload_session_finish(chunk, cursor, commit)
                                else:
                                    dbx.files_upload_session_append_v2(chunk, cursor)
                                    cursor.offset = f.tell()
                                break  # success
                            except Exception as e:
                                print(f"⚠️  Chunk upload failed (attempt {retry+1}): {e}")
                                time.sleep(RETRY_DELAY)
                                if retry == MAX_RETRIES - 1:
                                    raise e
                        uploaded_bytes = f.tell()
                        pbar.update(len(chunk))
                        time.sleep(1)  # avoid throttling
            print(f"✅ Uploaded {os.path.basename(local_path)} completely")
            return

        except Exception as e:
            print(f"❌ Upload attempt {attempt+1} failed for {os.path.basename(local_path)}: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"🔁 Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"🚨 Failed after {MAX_RETRIES} attempts. Try resuming later.")
                return

# --- Upload folder ---
def upload_folder(local_folder, dropbox_folder):
    files = [f for f in os.listdir(local_folder) if os.path.isfile(os.path.join(local_folder, f))]
    print(f"🚀 Uploading {len(files)} files from {local_folder} → {dropbox_folder}")
    for fname in files:
        local_path = os.path.join(local_folder, fname)
        dropbox_path = f"{dropbox_folder}/{fname}"
        print(f"⬆️  Uploading {fname} ...")
        upload_large_file(local_path, dropbox_path)
    print("🎉 All files uploaded successfully (or resumed where possible)!")

# --- Run upload ---
upload_folder(LOCAL_MODEL_DIR, DROPBOX_MODEL_DIR)
