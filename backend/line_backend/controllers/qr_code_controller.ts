import express, { Request, Response, NextFunction } from 'express';
import { QRCodeService } from '../services/qr_code_service';
import { QRCodeModel } from '../models/qr_code_model';
import { Logger } from '../../utils/logger';
import { authenticateUser, authorizeRole } from '../../middleware/auth';

class QRCodeController {
  private qrCodeService: QRCodeService;
  private logger: Logger;

  constructor(qrCodeService: QRCodeService, logger: Logger) {
    this.qrCodeService = qrCodeService;
    this.logger = logger;
  }

  /**
   * QRコードを生成する
   */
  public generateQRCode = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { userId, companyId, type, creditValue, expiryDate } = req.body;

      this.logger.info(`Generating QR code for user: ${userId}, company: ${companyId}`);
      
      const qrCode = await this.qrCodeService.generateQRCode({
        userId,
        companyId,
        type,
        creditValue,
        expiryDate
      });

      res.status(201).json({
        success: true,
        message: 'QRコードが正常に生成されました',
        data: qrCode
      });
    } catch (error) {
      this.logger.error(`QRコード生成エラー: ${error.message}`);
      next(error);
    }
  }

  /**
   * QRコードを検証する
   */
  public verifyQRCode = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { qrCodeId, staffId } = req.body;

      this.logger.info(`Verifying QR code: ${qrCodeId} by staff: ${staffId}`);
      
      const verificationResult = await this.qrCodeService.verifyQRCode(qrCodeId, staffId);

      if (!verificationResult.isValid) {
        return res.status(400).json({
          success: false,
          message: verificationResult.message || 'QRコードが無効です'
        });
      }

      res.status(200).json({
        success: true,
        message: 'QRコードが正常に検証されました',
        data: verificationResult
      });
    } catch (error) {
      this.logger.error(`QRコード検証エラー: ${error.message}`);
      next(error);
    }
  }

  /**
   * QRコードを無効化する
   */
  public invalidateQRCode = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { qrCodeId, reason } = req.body;
      const { userId } = req.user; // 認証済みユーザー情報から取得

      this.logger.info(`Invalidating QR code: ${qrCodeId} by user: ${userId}, reason: ${reason}`);
      
      const result = await this.qrCodeService.invalidateQRCode(qrCodeId, userId, reason);

      res.status(200).json({
        success: true,
        message: 'QRコードが正常に無効化されました',
        data: result
      });
    } catch (error) {
      this.logger.error(`QRコード無効化エラー: ${error.message}`);
      next(error);
    }
  }

  /**
   * QRコード情報を取得する
   */
  public getQRCodeInfo = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { qrCodeId } = req.params;

      this.logger.info(`Fetching QR code info: ${qrCodeId}`);
      
      const qrCodeInfo = await this.qrCodeService.getQRCodeInfo(qrCodeId);

      if (!qrCodeInfo) {
        return res.status(404).json({
          success: false,
          message: '指定されたQRコードが見つかりません'
        });
      }

      res.status(200).json({
        success: true,
        data: qrCodeInfo
      });
    } catch (error) {
      this.logger.error(`QRコード情報取得エラー: ${error.message}`);
      next(error);
    }
  }

  /**
   * ユーザーに関連するQRコード一覧を取得する
   */
  public getUserQRCodes = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { userId } = req.params;
      const { status, page = 1, limit = 10 } = req.query;

      this.logger.info(`Fetching QR codes for user: ${userId}, status: ${status}, page: ${page}, limit: ${limit}`);
      
      const qrCodes = await this.qrCodeService.getUserQRCodes(
        userId, 
        status as string, 
        Number(page), 
        Number(limit)
      );

      res.status(200).json({
        success: true,
        data: qrCodes
      });
    } catch (error) {
      this.logger.error(`ユーザーQRコード一覧取得エラー: ${error.message}`);
      next(error);
    }
  }

  /**
   * 企業に関連するQRコード一覧を取得する
   */
  public getCompanyQRCodes = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { companyId } = req.params;
      const { status, startDate, endDate, page = 1, limit = 10 } = req.query;

      this.logger.info(`Fetching QR codes for company: ${companyId}`);
      
      const qrCodes = await this.qrCodeService.getCompanyQRCodes(
        companyId,
        {
          status: status as string,
          startDate: startDate ? new Date(startDate as string) : undefined,
          endDate: endDate ? new Date(endDate as string) : undefined,
          page: Number(page),
          limit: Number(limit)
        }
      );

      res.status(200).json({
        success: true,
        data: qrCodes
      });
    } catch (error) {
      this.logger.error(`企業QRコード一覧取得エラー: ${error.message}`);
      next(error);
    }
  }

  /**
   * ルーターを設定する
   */
  public setupRoutes(router: express.Router): void {
    // QRコード生成 (認証とVC企業管理者または管理者権限が必要)
    router.post(
      '/qr-codes',
      authenticateUser,
      authorizeRole(['admin', 'vc_admin']),
      this.generateQRCode
    );

    // QRコード検証 (認証とスタッフ権限が必要)
    router.post(
      '/qr-codes/verify',
      authenticateUser,
      authorizeRole(['admin', 'staff']),
      this.verifyQRCode
    );

    // QRコード無効化 (認証と管理者権限または所有者が必要)
    router.post(
      '/qr-codes/:qrCodeId/invalidate',
      authenticateUser,
      this.invalidateQRCode
    );

    // QRコード情報取得 (認証必要)
    router.get(
      '/qr-codes/:qrCodeId',
      authenticateUser,
      this.getQRCodeInfo
    );

    // ユーザーQRコード一覧取得 (認証と管理者権限または本人が必要)
    router.get(
      '/users/:userId/qr-codes',
      authenticateUser,
      this.getUserQRCodes
    );

    // 企業QRコード一覧取得 (認証と管理者権限または企業管理者が必要)
    router.get(
      '/companies/:companyId/qr-codes',
      authenticateUser,
      authorizeRole(['admin', 'vc_admin', 'company_admin']),
      this.getCompanyQRCodes
    );
  }
}

export default QRCodeController;