import { Request, Response, NextFunction } from 'express';
import { CreditModel } from '../models/credit_model';
import { CreditService } from '../services/credit_service';
import { logger } from '../../utils/logger';
import { AuthError, ValidationError, NotFoundError } from '../../utils/errors';

class CreditController {
  private creditService: CreditService;

  constructor() {
    this.creditService = new CreditService();
  }

  /**
   * VCからポートフォリオ企業へのクレジット割り当て
   */
  public async allocateCredits(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      // リクエストの検証
      const { vcId, portfolioId, amount, expirationDate } = req.body;
      
      if (!vcId || !portfolioId || !amount) {
        throw new ValidationError('必須パラメータが不足しています');
      }

      // 認可チェック - リクエスト元がVC管理者か確認
      if (req.user.role !== 'vc_admin' || req.user.vcId !== vcId) {
        throw new AuthError('クレジット割り当ての権限がありません');
      }

      logger.info(`クレジット割り当て開始: ${vcId} -> ${portfolioId}, ${amount}クレジット`);
      
      const result = await this.creditService.allocateCredits(vcId, portfolioId, amount, expirationDate);
      
      logger.info(`クレジット割り当て成功: ID=${result.id}`);
      
      res.status(201).json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error(`クレジット割り当て失敗: ${error.message}`, { error });
      next(error);
    }
  }

  /**
   * クレジットの使用
   */
  public async useCredits(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { portfolioId, amount, serviceId, notes } = req.body;
      
      if (!portfolioId || !amount || !serviceId) {
        throw new ValidationError('必須パラメータが不足しています');
      }

      // 認可チェック - 適切な権限を持つユーザーかチェック
      if (!['staff', 'portfolio_admin'].includes(req.user.role)) {
        throw new AuthError('クレジット使用の権限がありません');
      }

      logger.info(`クレジット使用開始: ${portfolioId}, ${amount}クレジット, サービス=${serviceId}`);
      
      const transaction = await this.creditService.useCredits(portfolioId, amount, serviceId, notes, req.user.id);
      
      logger.info(`クレジット使用成功: トランザクションID=${transaction.id}`);
      
      res.status(200).json({
        success: true,
        data: transaction
      });
    } catch (error) {
      logger.error(`クレジット使用失敗: ${error.message}`, { error });
      next(error);
    }
  }

  /**
   * クレジット残高の確認
   */
  public async getBalance(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { portfolioId } = req.params;
      
      if (!portfolioId) {
        throw new ValidationError('企業IDが指定されていません');
      }

      // 認可チェック - 自社または関連VCの情報のみ参照可能
      const userHasAccess = 
        req.user.portfolioId === portfolioId || 
        (req.user.role === 'vc_admin' && await this.creditService.isPortfolioManagedByVC(portfolioId, req.user.vcId));
      
      if (!userHasAccess) {
        throw new AuthError('この企業のクレジット情報へのアクセス権限がありません');
      }

      logger.info(`クレジット残高確認: ${portfolioId}`);
      
      const balance = await this.creditService.getBalance(portfolioId);
      
      res.status(200).json({
        success: true,
        data: balance
      });
    } catch (error) {
      logger.error(`クレジット残高確認失敗: ${error.message}`, { error });
      next(error);
    }
  }

  /**
   * クレジット履歴の取得
   */
  public async getTransactionHistory(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { portfolioId } = req.params;
      const { startDate, endDate, page = 1, limit = 20 } = req.query;
      
      if (!portfolioId) {
        throw new ValidationError('企業IDが指定されていません');
      }

      // 認可チェック
      const userHasAccess = 
        req.user.portfolioId === portfolioId || 
        (req.user.role === 'vc_admin' && await this.creditService.isPortfolioManagedByVC(portfolioId, req.user.vcId));
      
      if (!userHasAccess) {
        throw new AuthError('この企業のクレジット履歴へのアクセス権限がありません');
      }

      logger.info(`クレジット履歴取得: ${portfolioId}, 期間=${startDate || 'all'} to ${endDate || 'all'}, ページ=${page}`);
      
      const history = await this.creditService.getTransactionHistory(
        portfolioId, 
        startDate ? new Date(startDate as string) : undefined,
        endDate ? new Date(endDate as string) : undefined,
        Number(page),
        Number(limit)
      );
      
      res.status(200).json({
        success: true,
        data: history.transactions,
        pagination: {
          total: history.total,
          page: Number(page),
          limit: Number(limit),
          pages: Math.ceil(history.total / Number(limit))
        }
      });
    } catch (error) {
      logger.error(`クレジット履歴取得失敗: ${error.message}`, { error });
      next(error);
    }
  }

  /**
   * 期限切れクレジットの管理（定期実行用）
   */
  public async handleExpiredCredits(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      // 管理者のみ実行可能
      if (req.user.role !== 'admin' && req.user.role !== 'system') {
        throw new AuthError('この操作を実行する権限がありません');
      }

      logger.info('期限切れクレジット処理開始');
      
      const result = await this.creditService.handleExpiredCredits();
      
      logger.info(`期限切れクレジット処理完了: ${result.expiredCount}件のクレジットを期限切れとしてマーク`);
      
      res.status(200).json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error(`期限切れクレジット処理失敗: ${error.message}`, { error });
      next(error);
    }
  }

  /**
   * クレジットの詳細情報取得
   */
  public async getCreditDetails(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { creditId } = req.params;
      
      if (!creditId) {
        throw new ValidationError('クレジットIDが指定されていません');
      }

      logger.info(`クレジット詳細取得: ${creditId}`);
      
      const creditDetails = await this.creditService.getCreditById(creditId);
      
      if (!creditDetails) {
        throw new NotFoundError('指定されたクレジットが見つかりません');
      }

      // 認可チェック
      const userHasAccess = 
        req.user.portfolioId === creditDetails.portfolioId || 
        (req.user.role === 'vc_admin' && req.user.vcId === creditDetails.vcId) ||
        req.user.role === 'admin';
      
      if (!userHasAccess) {
        throw new AuthError('このクレジット情報へのアクセス権限がありません');
      }
      
      res.status(200).json({
        success: true,
        data: creditDetails
      });
    } catch (error) {
      logger.error(`クレジット詳細取得失敗: ${error.message}`, { error });
      next(error);
    }
  }
}

export default new CreditController();