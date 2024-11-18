import React, { useState, useRef } from 'react';
import { Button, Grid, Typography, DropzoneArea } from '@mui/material';
import DataService from '../services/DataService';

const DataUpload = () => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const fileInputRef = useRef(null);

  const handleFileChange = (files) => {
    setUploadedFiles(files);
  };

  const handleUpload = async () => {
    try {
      const formData = new FormData();
      uploadedFiles.forEach((file) => {
        formData.append('files', file);
      });
      await DataService.uploadData(formData);
      // アップロード成功時の処理 (例: メッセージ表示, ファイルリストリセット)
      console.log('Data uploaded successfully.');
      setUploadedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = null;
      }
    } catch (error) {
      // アップロード失敗時の処理 (例: エラーメッセージ表示)
      console.error('Failed to upload data:', error);
    }
  };

  return (
    <Grid container spacing={2} justifyContent="center">
      <Grid item xs={12}>
        <Typography variant="h6" align="center" gutterBottom>
          Data Upload
        </Typography>
      </Grid>
      <Grid item xs={12} md={6}>
        <DropzoneArea
          onChange={handleFileChange}
          acceptedFiles={['.csv', '.xlsx', '.pdf']}
          showPreviews={true}
          showPreviewsInDropzone={false}
          filesLimit={3} // ファイルアップロード数制限
          dropzoneText="Drag and drop files here or click"
          inputRef={fileInputRef}
        />
      </Grid>
      <Grid item xs={12} align="center">
        <Button variant="contained" color="primary" onClick={handleUpload} disabled={uploadedFiles.length === 0}>
          Upload
        </Button>
      </Grid>
    </Grid>
  );
};

export default DataUpload;