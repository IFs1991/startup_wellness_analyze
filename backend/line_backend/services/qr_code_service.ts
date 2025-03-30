import QRCode from 'qrcode';
import { v4 as uuidv4 } from 'uuid';
import { QRCodeModel, QRCodeStatus } from '../models/qr_code_model';
import logger from '../../utils/logger';

interface QRCodeOptions {
  userId: string;
  companyId: string;
  type: 'single' | 'persistent';
  creditValue: number;
  expirationDate?: Date;
}

interface QRVerificationResult {
  isValid: boolean;
  message: string;
  qrCode?: QRCodeModel;
}

class QRCodeService {
  /**
   * QRコードを生成し、データベースに保存する
   */
  async generateQRCode(options: QRCodeOptions): Promise<{ qrCodeId: string; qrCodeImage: string }> {
    try {
      // ユニークなIDを生成
      const qrCodeId = uuidv4();
      
      // QRコードモデルの作成
      const qrCode = new QRCodeModel({
        id: qrCodeId,
        userId: options.userId,
        companyId: options.companyId,
        type: options.type,
        creditValue: options.creditValue,
        createdAt: new Date(),
        expirationDate: options.expirationDate || this.calculateDefaultExpiration(options.type),
        status: QRCodeStatus.ACTIVE
      });
      
      // データベースに保存
      await qrCode.save();
      
      // QRコード画像の生成
      const qrCodeData = JSON.stringify({
        id: qrCodeId,
        timestamp: new Date().getTime()
      });
      
      const qrCodeImage = await QRCode.toDataURL(qrCodeData, {
        errorCorrectionLevel: 'H',
        margin: 1,
        width: 300
      });
      
      logger.info(`QRコードが生成されました: ${qrCodeId}`);
      
      return { qrCodeId, qrCodeImage };
    } catch (error) {
      logger.error(`QRコード生成中にエラーが発生しました: ${error.message}`);
      throw new Error(`QRコード生成に失敗しました: ${error.message}`);
    }
  }

  /**
   * QRコードの有効性を検証する
   */
  async verifyQRCode(qrCodeId: string): Promise<QRVerificationResult> {
    try {
      // データベースからQRコードを取得
      const qrCode = await QRCodeModel.findById(qrCodeId);
      
      if (!qrCode) {
        logger.warn(`QRコードが見つかりません: ${qrCodeId}`);
        return { isValid: false, message: 'QRコードが見つかりません' };
      }
      
      // ステータスの確認
      if (qrCode.status !== QRCodeStatus.ACTIVE) {
        logger.warn(`QRコードは既に使用済みまたは無効です: ${qrCodeId}`);
        return { isValid: false, message: 'QRコードは既に使用済みまたは無効です' };
      }
      
      // 有効期限の確認
      if (qrCode.expirationDate && new Date() > qrCode.expirationDate) {
        // 有効期限切れの場合はステータスを更新
        qrCode.status = QRCodeStatus.EXPIRED;
        await qrCode.save();
        
        logger.warn(`QRコードの有効期限が切れています: ${qrCodeId}`);
        return { isValid: false, message: 'QRコードの有効期限が切れています' };
      }
      
      logger.info(`QRコードの検証に成功しました: ${qrCodeId}`);
      return { isValid: true, message: 'QRコードは有効です', qrCode };
    } catch (error) {
      logger.error(`QRコード検証中にエラーが発生しました: ${error.message}`);
      throw new Error(`QRコード検証に失敗しました: ${error.message}`);
    }
  }

  /**
   * QRコードを使用済みにする
   */
  async markQRCodeAsUsed(qrCodeId: string, staffId: string): Promise<boolean> {
    try {
      // データベースからQRコードを取得
      const qrCode = await QRCodeModel.findById(qrCodeId);
      
      if (!qrCode) {
        logger.warn(`QRコードが見つかりません: ${qrCodeId}`);
        return false;
      }
      
      // 既に使用済みか確認
      if (qrCode.status === QRCodeStatus.USED) {
        logger.warn(`QRコードは既に使用済みです: ${qrCodeId}`);
        return false;
      }
      
      // 有効期限切れか確認
      if (qrCode.expirationDate && new Date() > qrCode.expirationDate) {
        qrCode.status = QRCodeStatus.EXPIRED;
        await qrCode.save();
        
        logger.warn(`QRコードの有効期限が切れています: ${qrCodeId}`);
        return false;
      }
      
      // QRコードを使用済みに更新
      qrCode.status = QRCodeStatus.USED;
      qrCode.usedAt = new Date();
      qrCode.usedBy = staffId;
      await qrCode.save();
      
      logger.info(`QRコードが使用済みになりました: ${qrCodeId} (スタッフ: ${staffId})`);
      return true;
    } catch (error) {
      logger.error(`QRコード使用処理中にエラーが発生しました: ${error.message}`);
      throw new Error(`QRコードの使用処理に失敗しました: ${error.message}`);
    }
  }

  /**
   * QRコードを無効化する
   */
  async invalidateQRCode(qrCodeId: string, reason: string): Promise<boolean> {
    try {
      // データベースからQRコードを取得
      const qrCode = await QRCodeModel.findById(qrCodeId);
      
      if (!qrCode) {
        logger.warn(`QRコードが見つかりません: ${qrCodeId}`);
        return false;
      }
      
      // 既に無効か使用済みか確認
      if (qrCode.status === QRCodeStatus.CANCELLED || qrCode.status === QRCodeStatus.USED) {
        logger.warn(`QRコードは既に無効または使用済みです: ${qrCodeId}`);
        return false;
      }
      
      // QRコードを無効に更新
      qrCode.status = QRCodeStatus.CANCELLED;
      qrCode.cancellationReason = reason;
      await qrCode.save();
      
      logger.info(`QRコードが無効化されました: ${qrCodeId} (理由: ${reason})`);
      return true;
    } catch (error) {
      logger.error(`QRコード無効化中にエラーが発生しました: ${error.message}`);
      throw new Error(`QRコードの無効化に失敗しました: ${error.message}`);
    }
  }

  /**
   * QRコード情報を取得する
   */
  async getQRCodeInfo(qrCodeId: string): Promise<QRCodeModel | null> {
    try {
      // データベースからQRコードを取得
      const qrCode = await QRCodeModel.findById(qrCodeId);
      
      if (!qrCode) {
        logger.warn(`QRコードが見つかりません: ${qrCodeId}`);
        return null;
      }
      
      logger.info(`QRコード情報が取得されました: ${qrCodeId}`);
      return qrCode;
    } catch (error) {
      logger.error(`QRコード情報取得中にエラーが発生しました: ${error.message}`);
      throw new Error(`QRコード情報の取得に失敗しました: ${error.message}`);
    }
  }

  /**
   * ユーザーに関連するすべてのQRコードを取得する
   */
  async getQRCodesByUserId(userId: string): Promise<QRCodeModel[]> {
    try {
      // データベースからユーザーに関連するQRコードを取得
      const qrCodes = await QRCodeModel.findByUserId(userId);
      
      logger.info(`ユーザー(${userId})のQRコードが取得されました: ${qrCodes.length}件`);
      return qrCodes;
    } catch (error) {
      logger.error(`ユーザーQRコード取得中にエラーが発生しました: ${error.message}`);
      throw new Error(`ユーザーのQRコード取得に失敗しました: ${error.message}`);
    }
  }

  /**
   * 企業に関連するすべてのQRコードを取得する
   */
  async getQRCodesByCompanyId(companyId: string): Promise<QRCodeModel[]> {
    try {
      // データベースから企業に関連するQRコードを取得
      const qrCodes = await QRCodeModel.findByCompanyId(companyId);
      
      logger.info(`企業(${companyId})のQRコードが取得されました: ${qrCodes.length}件`);
      return qrCodes;
    } catch (error) {
      logger.error(`企業QRコード取得中にエラーが発生しました: ${error.message}`);
      throw new Error(`企業のQRコード取得に失敗しました: ${error.message}`);
    }
  }

  /**
   * QRコードの有効期限を更新する
   */
  async updateQRCodeExpiration(qrCodeId: string, newExpirationDate: Date): Promise<boolean> {
    try {
      // データベースからQRコードを取得
      const qrCode = await QRCodeModel.findById(qrCodeId);
      
      if (!qrCode) {
        logger.warn(`QRコードが見つかりません: ${qrCodeId}`);
        return false;
      }
      
      // 既に使用済みか確認
      if (qrCode.status !== QRCodeStatus.ACTIVE) {
        logger.warn(`有効でないQRコードの有効期限は更新できません: ${qrCodeId}`);
        return false;
      }
      
      // 有効期限を更新
      qrCode.expirationDate = newExpirationDate;
      await qrCode.save();
      
      logger.info(`QRコードの有効期限が更新されました: ${qrCodeId}`);
      return true;
    } catch (error) {
      logger.error(`QRコード有効期限更新中にエラーが発生しました: ${error.message}`);
      throw new Error(`QRコードの有効期限更新に失敗しました: ${error.message}`);
    }
  }

  /**
   * QRコードタイプに基づいてデフォルトの有効期限を計算する
   */
  private calculateDefaultExpiration(type: 'single' | 'persistent'): Date {
    const now = new Date();
    
    if (type === 'single') {
      // 単回使用の場合は24時間有効
      return new Date(now.setHours(now.getHours() + 24));
    } else {
      // 永続的な場合は1年間有効
      return new Date(now.setFullYear(now.getFullYear() + 1));
    }
  }
}

export default new QRCodeService();