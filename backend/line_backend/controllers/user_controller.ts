import { Request, Response, NextFunction } from 'express';
import { UserService } from '../services/user_service';
import { User } from '../models/user_model';
import logger from '../../utils/logger';

export class UserController {
  private userService: UserService;

  constructor() {
    this.userService = new UserService();
  }

  /**
   * ユーザー登録
   */
  public async registerUser(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      logger.info('ユーザー登録処理開始', { email: req.body.email });
      const userData = req.body as User;
      const newUser = await this.userService.createUser(userData);
      
      res.status(201).json({
        success: true,
        message: 'ユーザーが正常に登録されました',
        data: newUser
      });
      logger.info('ユーザー登録処理完了', { userId: newUser.id });
    } catch (error) {
      logger.error('ユーザー登録処理エラー', { error: error.message, stack: error.stack });
      next(error);
    }
  }

  /**
   * ユーザー情報取得（単一ユーザー）
   */
  public async getUserById(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const userId = req.params.id;
      logger.info('ユーザー情報取得処理開始', { userId });
      
      const user = await this.userService.getUserById(userId);
      if (!user) {
        res.status(404).json({
          success: false,
          message: 'ユーザーが見つかりません'
        });
        return;
      }

      res.status(200).json({
        success: true,
        data: user
      });
      logger.info('ユーザー情報取得処理完了', { userId });
    } catch (error) {
      logger.error('ユーザー情報取得処理エラー', { error: error.message, stack: error.stack });
      next(error);
    }
  }

  /**
   * ユーザー情報取得（LINE IDによる取得）
   */
  public async getUserByLineId(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const lineUserId = req.params.lineUserId;
      logger.info('LINE IDによるユーザー情報取得処理開始', { lineUserId });
      
      const user = await this.userService.getUserByLineId(lineUserId);
      if (!user) {
        res.status(404).json({
          success: false,
          message: 'ユーザーが見つかりません'
        });
        return;
      }

      res.status(200).json({
        success: true,
        data: user
      });
      logger.info('LINE IDによるユーザー情報取得処理完了', { lineUserId });
    } catch (error) {
      logger.error('LINE IDによるユーザー情報取得処理エラー', { error: error.message, stack: error.stack });
      next(error);
    }
  }

  /**
   * ユーザー一覧取得
   */
  public async getAllUsers(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      logger.info('ユーザー一覧取得処理開始');
      const users = await this.userService.getAllUsers();
      
      res.status(200).json({
        success: true,
        count: users.length,
        data: users
      });
      logger.info('ユーザー一覧取得処理完了', { count: users.length });
    } catch (error) {
      logger.error('ユーザー一覧取得処理エラー', { error: error.message, stack: error.stack });
      next(error);
    }
  }

  /**
   * ユーザー情報更新
   */
  public async updateUser(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const userId = req.params.id;
      const userData = req.body as Partial<User>;
      
      logger.info('ユーザー情報更新処理開始', { userId });
      
      // リクエストユーザーとターゲットユーザーの権限チェック
      if (!this.userService.canModifyUser(req.user, userId)) {
        res.status(403).json({
          success: false,
          message: 'このユーザー情報を更新する権限がありません'
        });
        return;
      }
      
      const updatedUser = await this.userService.updateUser(userId, userData);
      if (!updatedUser) {
        res.status(404).json({
          success: false,
          message: 'ユーザーが見つかりません'
        });
        return;
      }

      res.status(200).json({
        success: true,
        message: 'ユーザー情報が正常に更新されました',
        data: updatedUser
      });
      logger.info('ユーザー情報更新処理完了', { userId });
    } catch (error) {
      logger.error('ユーザー情報更新処理エラー', { error: error.message, stack: error.stack });
      next(error);
    }
  }

  /**
   * ユーザー削除
   */
  public async deleteUser(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const userId = req.params.id;
      logger.info('ユーザー削除処理開始', { userId });
      
      // リクエストユーザーとターゲットユーザーの権限チェック
      if (!this.userService.canModifyUser(req.user, userId)) {
        res.status(403).json({
          success: false,
          message: 'このユーザーを削除する権限がありません'
        });
        return;
      }
      
      const deleted = await this.userService.deleteUser(userId);
      if (!deleted) {
        res.status(404).json({
          success: false,
          message: 'ユーザーが見つかりません'
        });
        return;
      }

      res.status(200).json({
        success: true,
        message: 'ユーザーが正常に削除されました'
      });
      logger.info('ユーザー削除処理完了', { userId });
    } catch (error) {
      logger.error('ユーザー削除処理エラー', { error: error.message, stack: error.stack });
      next(error);
    }
  }

  /**
   * ユーザーLINE連携
   */
  public async linkLineAccount(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const userId = req.params.id;
      const { lineUserId } = req.body;
      
      logger.info('ユーザーLINE連携処理開始', { userId, lineUserId });
      
      const updatedUser = await this.userService.linkLineAccount(userId, lineUserId);
      if (!updatedUser) {
        res.status(404).json({
          success: false,
          message: 'ユーザーが見つかりません'
        });
        return;
      }

      res.status(200).json({
        success: true,
        message: 'LINEアカウントが正常に連携されました',
        data: updatedUser
      });
      logger.info('ユーザーLINE連携処理完了', { userId, lineUserId });
    } catch (error) {
      logger.error('ユーザーLINE連携処理エラー', { error: error.message, stack: error.stack });
      next(error);
    }
  }

  /**
   * ユーザー認証（ログイン）
   */
  public async loginUser(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { email, password } = req.body;
      logger.info('ユーザー認証処理開始', { email });
      
      const { user, token } = await this.userService.authenticateUser(email, password);
      
      res.status(200).json({
        success: true,
        message: 'ログインに成功しました',
        token,
        data: user
      });
      logger.info('ユーザー認証処理完了', { userId: user.id });
    } catch (error) {
      logger.error('ユーザー認証処理エラー', { error: error.message });
      
      // 認証エラーの場合は401を返す
      if (error.message === 'Invalid credentials') {
        res.status(401).json({
          success: false,
          message: 'メールアドレスまたはパスワードが正しくありません'
        });
        return;
      }
      
      next(error);
    }
  }
}