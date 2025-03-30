import { CreditModel, CreditTransaction, CreditTransactionType } from '../models/credit_model';
import { Logger } from '../../utils/logger';

/**
 * クレジット管理に関するビジネスロジックを実装するサービスクラス
 */
export class CreditService {
  private logger: Logger;

  constructor(private creditModel: CreditModel) {
    this.logger = new Logger('CreditService');
  }

  /**
   * VCからポートフォリオ企業にクレジットを割り当てる
   */
  async assignCredit(
    vcId: string,
    portfolioCompanyId: string,
    creditAmount: number,
    expirationDate: Date
  ): Promise<boolean> {
    try {
      // 入力値のバリデーション
      if (creditAmount <= 0) {
        throw new Error('Credit amount must be greater than zero');
      }
      
      if (expirationDate <= new Date()) {
        throw new Error('Expiration date must be in the future');
      }

      // クレジット割り当て処理
      const result = await this.creditModel.createCredit({
        vcId,
        portfolioCompanyId,
        totalCredits: creditAmount,
        usedCredits: 0,
        remainingCredits: creditAmount,
        expirationDate,
        lastUpdated: new Date()
      });

      // クレジット割り当て履歴の記録
      await this.recordCreditTransaction(
        result.id,
        CreditTransactionType.ASSIGN,
        creditAmount,
        `${creditAmount} credits assigned to portfolio company ${portfolioCompanyId}`
      );

      this.logger.info(`Credits successfully assigned to portfolio company ${portfolioCompanyId}`);
      return true;
    } catch (error) {
      this.logger.error(`Failed to assign credits: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * クレジットを使用する
   */
  async useCredit(
    creditId: string,
    amountToUse: number,
    userId: string,
    serviceId: string
  ): Promise<boolean> {
    try {
      // 入力値のバリデーション
      if (amountToUse <= 0) {
        throw new Error('Amount to use must be greater than zero');
      }

      // クレジット情報の取得
      const creditInfo = await this.creditModel.getCreditById(creditId);
      if (!creditInfo) {
        throw new Error(`Credit with ID ${creditId} not found`);
      }

      // 有効期限のチェック
      if (new Date() > creditInfo.expirationDate) {
        throw new Error('Credit has expired');
      }

      // 残高のチェック
      if (creditInfo.remainingCredits < amountToUse) {
        throw new Error('Insufficient credit balance');
      }

      // クレジット使用処理
      const updatedRemainingCredits = creditInfo.remainingCredits - amountToUse;
      const updatedUsedCredits = creditInfo.usedCredits + amountToUse;
      
      await this.creditModel.updateCredit(creditId, {
        usedCredits: updatedUsedCredits,
        remainingCredits: updatedRemainingCredits,
        lastUpdated: new Date()
      });

      // クレジット使用履歴の記録
      await this.recordCreditTransaction(
        creditId,
        CreditTransactionType.USE,
        amountToUse,
        `${amountToUse} credits used by user ${userId} for service ${serviceId}`
      );

      this.logger.info(`Credits successfully used: ${amountToUse} from credit ID ${creditId}`);
      return true;
    } catch (error) {
      this.logger.error(`Failed to use credits: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * クレジット残高を取得する
   */
  async getCreditBalance(creditId: string): Promise<number> {
    try {
      const creditInfo = await this.creditModel.getCreditById(creditId);
      if (!creditInfo) {
        throw new Error(`Credit with ID ${creditId} not found`);
      }

      // 有効期限切れの場合は0を返す
      if (new Date() > creditInfo.expirationDate) {
        this.logger.warn(`Credit with ID ${creditId} has expired`);
        return 0;
      }

      return creditInfo.remainingCredits;
    } catch (error) {
      this.logger.error(`Failed to get credit balance: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * ポートフォリオ企業の全クレジット残高を取得する
   */
  async getTotalCreditBalanceForCompany(portfolioCompanyId: string): Promise<number> {
    try {
      const credits = await this.creditModel.getCreditsByPortfolioCompany(portfolioCompanyId);
      const currentDate = new Date();
      
      // 有効期限内のクレジットのみを合計
      const totalBalance = credits.reduce((total, credit) => {
        if (currentDate <= credit.expirationDate) {
          return total + credit.remainingCredits;
        }
        return total;
      }, 0);

      return totalBalance;
    } catch (error) {
      this.logger.error(`Failed to get total credit balance for company: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * クレジット履歴を記録する
   */
  private async recordCreditTransaction(
    creditId: string, 
    type: CreditTransactionType, 
    amount: number, 
    description: string
  ): Promise<void> {
    try {
      const transaction: CreditTransaction = {
        creditId,
        type,
        amount,
        description,
        timestamp: new Date()
      };

      await this.creditModel.createCreditTransaction(transaction);
    } catch (error) {
      this.logger.error(`Failed to record credit transaction: ${error.message}`, error);
      // 履歴記録失敗はスローせずログのみ
    }
  }

  /**
   * クレジット履歴を取得する
   */
  async getCreditTransactionHistory(creditId: string): Promise<CreditTransaction[]> {
    try {
      return await this.creditModel.getCreditTransactions(creditId);
    } catch (error) {
      this.logger.error(`Failed to get credit transaction history: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * 有効期限が近いクレジットを取得する
   */
  async getExpiringCredits(daysToExpiration: number): Promise<any[]> {
    try {
      const expirationDate = new Date();
      expirationDate.setDate(expirationDate.getDate() + daysToExpiration);
      
      const currentDate = new Date();
      const expiringCredits = await this.creditModel.getCreditsExpiringBefore(expirationDate);
      
      // 既に期限切れでないものかつ残高があるものだけ返す
      return expiringCredits.filter(credit => 
        credit.expirationDate > currentDate && 
        credit.remainingCredits > 0
      );
    } catch (error) {
      this.logger.error(`Failed to get expiring credits: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * 期限切れクレジットを無効にする
   */
  async invalidateExpiredCredits(): Promise<number> {
    try {
      const currentDate = new Date();
      const expiredCredits = await this.creditModel.getCreditsExpiringBefore(currentDate);
      
      let invalidatedCount = 0;
      
      for (const credit of expiredCredits) {
        if (credit.remainingCredits > 0) {
          await this.recordCreditTransaction(
            credit.id,
            CreditTransactionType.EXPIRE,
            credit.remainingCredits,
            `${credit.remainingCredits} credits expired`
          );
          
          await this.creditModel.updateCredit(credit.id, {
            remainingCredits: 0,
            lastUpdated: new Date()
          });
          
          invalidatedCount++;
        }
      }
      
      this.logger.info(`Invalidated ${invalidatedCount} expired credits`);
      return invalidatedCount;
    } catch (error) {
      this.logger.error(`Failed to invalidate expired credits: ${error.message}`, error);
      throw error;
    }
  }
}