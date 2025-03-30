import { Request, Response, NextFunction } from 'express';
import { ReservationService } from '../services/reservation_service';
import { logger } from '../../utils/logger';
import { AuthMiddleware } from '../../middleware/auth_middleware';

export class ReservationController {
  private reservationService: ReservationService;
  private authMiddleware: AuthMiddleware;

  constructor() {
    this.reservationService = new ReservationService();
    this.authMiddleware = new AuthMiddleware();
  }

  /**
   * 予約を作成する
   */
  public createReservation = async (req: Request, res: Response): Promise<void> => {
    try {
      logger.info('ReservationController: Creating reservation', { userId: req.user?.id });
      
      const reservationData = req.body;
      const result = await this.reservationService.createReservation({
        ...reservationData,
        userId: req.user?.id
      });
      
      res.status(201).json({
        success: true,
        message: '予約が正常に作成されました',
        data: result
      });
    } catch (error) {
      logger.error('ReservationController: Error creating reservation', { error, userId: req.user?.id });
      this.handleError(error, res);
    }
  };

  /**
   * 予約を更新する
   */
  public updateReservation = async (req: Request, res: Response): Promise<void> => {
    try {
      const { id } = req.params;
      logger.info('ReservationController: Updating reservation', { id, userId: req.user?.id });
      
      // 予約の所有者確認
      await this.verifyReservationOwnership(id, req.user?.id, res);
      
      const updateData = req.body;
      const result = await this.reservationService.updateReservation(id, updateData);
      
      res.status(200).json({
        success: true,
        message: '予約が正常に更新されました',
        data: result
      });
    } catch (error) {
      logger.error('ReservationController: Error updating reservation', { 
        error, 
        reservationId: req.params.id,
        userId: req.user?.id 
      });
      this.handleError(error, res);
    }
  };

  /**
   * 予約をキャンセルする
   */
  public cancelReservation = async (req: Request, res: Response): Promise<void> => {
    try {
      const { id } = req.params;
      logger.info('ReservationController: Cancelling reservation', { id, userId: req.user?.id });
      
      // 予約の所有者確認
      await this.verifyReservationOwnership(id, req.user?.id, res);
      
      const result = await this.reservationService.cancelReservation(id);
      
      res.status(200).json({
        success: true,
        message: '予約が正常にキャンセルされました',
        data: result
      });
    } catch (error) {
      logger.error('ReservationController: Error cancelling reservation', { 
        error, 
        reservationId: req.params.id,
        userId: req.user?.id 
      });
      this.handleError(error, res);
    }
  };

  /**
   * 予約情報を取得する
   */
  public getReservation = async (req: Request, res: Response): Promise<void> => {
    try {
      const { id } = req.params;
      logger.info('ReservationController: Getting reservation', { id, userId: req.user?.id });
      
      const reservation = await this.reservationService.getReservationById(id);
      
      // 一般ユーザーは自分の予約のみ閲覧可能
      if (!req.user?.isAdmin && reservation.userId !== req.user?.id) {
        res.status(403).json({
          success: false,
          message: 'この予約の閲覧権限がありません'
        });
        return;
      }
      
      res.status(200).json({
        success: true,
        data: reservation
      });
    } catch (error) {
      logger.error('ReservationController: Error getting reservation', { 
        error, 
        reservationId: req.params.id,
        userId: req.user?.id 
      });
      this.handleError(error, res);
    }
  };

  /**
   * ユーザーの予約一覧を取得する
   */
  public getUserReservations = async (req: Request, res: Response): Promise<void> => {
    try {
      const userId = req.user?.id;
      logger.info('ReservationController: Getting user reservations', { userId });
      
      const reservations = await this.reservationService.getReservationsByUserId(userId);
      
      res.status(200).json({
        success: true,
        data: reservations
      });
    } catch (error) {
      logger.error('ReservationController: Error getting user reservations', { 
        error, 
        userId: req.user?.id 
      });
      this.handleError(error, res);
    }
  };

  /**
   * 管理者用: すべての予約を取得する
   */
  public getAllReservations = async (req: Request, res: Response): Promise<void> => {
    try {
      // 管理者権限チェック
      if (!req.user?.isAdmin) {
        res.status(403).json({
          success: false,
          message: '管理者権限が必要です'
        });
        return;
      }
      
      logger.info('ReservationController: Getting all reservations', { userId: req.user?.id });
      
      const { page = 1, limit = 10, status, startDate, endDate } = req.query;
      
      const reservations = await this.reservationService.getAllReservations({
        page: Number(page),
        limit: Number(limit),
        status: status as string,
        startDate: startDate as string,
        endDate: endDate as string
      });
      
      res.status(200).json({
        success: true,
        data: reservations
      });
    } catch (error) {
      logger.error('ReservationController: Error getting all reservations', { 
        error, 
        userId: req.user?.id,
        query: req.query
      });
      this.handleError(error, res);
    }
  };

  /**
   * 空き状況を確認する
   */
  public checkAvailability = async (req: Request, res: Response): Promise<void> => {
    try {
      const { date, treatmentId, staffId } = req.query;
      logger.info('ReservationController: Checking availability', { 
        date, treatmentId, staffId, userId: req.user?.id 
      });
      
      if (!date) {
        res.status(400).json({
          success: false,
          message: '日付の指定が必要です'
        });
        return;
      }
      
      const availableSlots = await this.reservationService.checkAvailability({
        date: date as string,
        treatmentId: treatmentId as string,
        staffId: staffId as string
      });
      
      res.status(200).json({
        success: true,
        data: availableSlots
      });
    } catch (error) {
      logger.error('ReservationController: Error checking availability', { 
        error, 
        userId: req.user?.id,
        query: req.query
      });
      this.handleError(error, res);
    }
  };

  /**
   * 予約所有者の確認を行う内部メソッド
   */
  private verifyReservationOwnership = async (
    reservationId: string, 
    userId: string, 
    res: Response
  ): Promise<boolean> => {
    const reservation = await this.reservationService.getReservationById(reservationId);
    
    // 管理者または予約の所有者のみ更新可能
    if (!reservation) {
      res.status(404).json({
        success: false,
        message: '予約が見つかりません'
      });
      return false;
    }
    
    if (reservation.userId !== userId && !res.locals.user?.isAdmin) {
      res.status(403).json({
        success: false,
        message: 'この予約を操作する権限がありません'
      });
      return false;
    }
    
    return true;
  };

  /**
   * エラーハンドリングの共通処理
   */
  private handleError = (error: any, res: Response): void => {
    if (error.name === 'ValidationError') {
      res.status(400).json({
        success: false,
        message: '入力データが不正です',
        errors: error.details
      });
    } else if (error.name === 'NotFoundError') {
      res.status(404).json({
        success: false,
        message: error.message || 'リソースが見つかりません'
      });
    } else if (error.name === 'ForbiddenError') {
      res.status(403).json({
        success: false,
        message: error.message || 'この操作を行う権限がありません'
      });
    } else if (error.name === 'ConflictError') {
      res.status(409).json({
        success: false,
        message: error.message || '予約の日時が重複しています'
      });
    } else {
      res.status(500).json({
        success: false,
        message: '内部サーバーエラーが発生しました'
      });
    }
  };

  /**
   * ルーターにコントローラーのメソッドを登録する
   */
  public routes() {
    return {
      // 予約作成
      'POST /reservations': [
        this.authMiddleware.authenticate,
        this.createReservation
      ],
      // 予約更新
      'PUT /reservations/:id': [
        this.authMiddleware.authenticate,
        this.updateReservation
      ],
      // 予約キャンセル
      'DELETE /reservations/:id': [
        this.authMiddleware.authenticate,
        this.cancelReservation
      ],
      // 個別予約取得
      'GET /reservations/:id': [
        this.authMiddleware.authenticate,
        this.getReservation
      ],
      // ユーザー予約一覧
      'GET /user/reservations': [
        this.authMiddleware.authenticate,
        this.getUserReservations
      ],
      // 管理者用全予約一覧
      'GET /admin/reservations': [
        this.authMiddleware.authenticate,
        this.authMiddleware.authorizeAdmin,
        this.getAllReservations
      ],
      // 空き状況確認
      'GET /reservations/availability': [
        this.checkAvailability
      ]
    };
  }
}

export default new ReservationController();