import { Reservation, ReservationStatus } from '../models/reservation_model';
import logger from '../../utils/logger';

export class ReservationService {
  /**
   * 新しい予約を作成する
   * @param userId ユーザーID
   * @param treatmentId 施術ID
   * @param datetime 予約日時
   * @param duration 施術時間（分）
   * @param staffId 担当者ID
   * @param notes 予約メモ
   * @returns 作成された予約
   */
  async createReservation(
    userId: string,
    treatmentId: string,
    datetime: Date,
    duration: number,
    staffId: string,
    notes?: string
  ): Promise<Reservation> {
    try {
      logger.info(`Creating reservation for user ${userId}, treatment ${treatmentId}`);
      
      // 空き状況の確認
      const isAvailable = await this.checkAvailability(staffId, datetime, duration);
      if (!isAvailable) {
        logger.warn(`Slot not available for staff ${staffId} at ${datetime}`);
        throw new Error('指定された時間枠は利用できません');
      }
      
      // 予約データの作成
      const reservation = new Reservation({
        userId,
        treatmentId,
        datetime,
        duration,
        staffId,
        status: ReservationStatus.RESERVED,
        notes,
        createdAt: new Date(),
        updatedAt: new Date()
      });
      
      // データベースに保存
      await reservation.save();
      
      logger.info(`Reservation created successfully with ID: ${reservation.id}`);
      return reservation;
    } catch (error) {
      logger.error(`Failed to create reservation: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * 予約情報を更新する
   * @param reservationId 予約ID
   * @param updateData 更新するデータ
   * @returns 更新された予約
   */
  async updateReservation(
    reservationId: string,
    updateData: {
      datetime?: Date;
      duration?: number;
      staffId?: string;
      notes?: string;
    }
  ): Promise<Reservation> {
    try {
      logger.info(`Updating reservation ${reservationId}`);
      
      // 予約の存在確認
      const reservation = await Reservation.findById(reservationId);
      if (!reservation) {
        logger.warn(`Reservation ${reservationId} not found`);
        throw new Error('予約が見つかりません');
      }
      
      // キャンセル済みの予約は変更不可
      if (reservation.status === ReservationStatus.CANCELLED) {
        logger.warn(`Cannot update cancelled reservation ${reservationId}`);
        throw new Error('キャンセルされた予約は変更できません');
      }
      
      // 日時または担当者変更の場合、空き状況を確認
      if (updateData.datetime || updateData.staffId) {
        const checkDatetime = updateData.datetime || reservation.datetime;
        const checkStaffId = updateData.staffId || reservation.staffId;
        const checkDuration = updateData.duration || reservation.duration;
        
        const isAvailable = await this.checkAvailability(
          checkStaffId, 
          checkDatetime, 
          checkDuration,
          reservationId // 自分自身の予約はスキップ
        );
        
        if (!isAvailable) {
          logger.warn(`Updated slot not available for staff ${checkStaffId} at ${checkDatetime}`);
          throw new Error('指定された時間枠は利用できません');
        }
      }
      
      // 予約データの更新
      Object.assign(reservation, {
        ...updateData,
        updatedAt: new Date()
      });
      
      await reservation.save();
      
      logger.info(`Reservation ${reservationId} updated successfully`);
      return reservation;
    } catch (error) {
      logger.error(`Failed to update reservation: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * 予約をキャンセルする
   * @param reservationId 予約ID
   * @param reason キャンセル理由
   * @returns キャンセルされた予約
   */
  async cancelReservation(reservationId: string, reason?: string): Promise<Reservation> {
    try {
      logger.info(`Cancelling reservation ${reservationId}`);
      
      // 予約の存在確認
      const reservation = await Reservation.findById(reservationId);
      if (!reservation) {
        logger.warn(`Reservation ${reservationId} not found`);
        throw new Error('予約が見つかりません');
      }
      
      // すでにキャンセル済みの場合
      if (reservation.status === ReservationStatus.CANCELLED) {
        logger.warn(`Reservation ${reservationId} is already cancelled`);
        return reservation;
      }
      
      // 完了済みの予約はキャンセル不可
      if (reservation.status === ReservationStatus.COMPLETED) {
        logger.warn(`Cannot cancel completed reservation ${reservationId}`);
        throw new Error('完了済みの予約はキャンセルできません');
      }
      
      // 予約のキャンセル
      reservation.status = ReservationStatus.CANCELLED;
      reservation.notes = reason ? `${reservation.notes || ''}\nキャンセル理由: ${reason}`.trim() : reservation.notes;
      reservation.updatedAt = new Date();
      
      await reservation.save();
      
      logger.info(`Reservation ${reservationId} cancelled successfully`);
      return reservation;
    } catch (error) {
      logger.error(`Failed to cancel reservation: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * 予約IDで予約情報を取得する
   * @param reservationId 予約ID
   * @returns 予約情報
   */
  async getReservationById(reservationId: string): Promise<Reservation> {
    try {
      logger.info(`Getting reservation by ID: ${reservationId}`);
      
      const reservation = await Reservation.findById(reservationId);
      if (!reservation) {
        logger.warn(`Reservation ${reservationId} not found`);
        throw new Error('予約が見つかりません');
      }
      
      return reservation;
    } catch (error) {
      logger.error(`Failed to get reservation: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * ユーザーIDで予約情報を取得する
   * @param userId ユーザーID
   * @param options 取得オプション（ステータス、日付範囲など）
   * @returns 予約情報の配列
   */
  async getReservationsByUserId(
    userId: string,
    options: {
      status?: ReservationStatus;
      startDate?: Date;
      endDate?: Date;
      limit?: number;
      offset?: number;
    } = {}
  ): Promise<Reservation[]> {
    try {
      logger.info(`Getting reservations for user ${userId}`);
      
      const query: any = { userId };
      
      // ステータスフィルタ
      if (options.status) {
        query.status = options.status;
      }
      
      // 日付範囲フィルタ
      if (options.startDate || options.endDate) {
        query.datetime = {};
        if (options.startDate) {
          query.datetime.$gte = options.startDate;
        }
        if (options.endDate) {
          query.datetime.$lte = options.endDate;
        }
      }
      
      // クエリ実行
      let reservationsQuery = Reservation.find(query)
        .sort({ datetime: 1 }); // 日付順にソート
      
      // ページネーション
      if (options.limit) {
        reservationsQuery = reservationsQuery.limit(options.limit);
        if (options.offset) {
          reservationsQuery = reservationsQuery.skip(options.offset);
        }
      }
      
      const reservations = await reservationsQuery.exec();
      
      logger.info(`Found ${reservations.length} reservations for user ${userId}`);
      return reservations;
    } catch (error) {
      logger.error(`Failed to get reservations for user: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * 特定の日の予約スケジュールを取得する
   * @param date 日付
   * @param staffId 担当者ID（省略可能）
   * @returns その日の予約一覧
   */
  async getDailySchedule(date: Date, staffId?: string): Promise<Reservation[]> {
    try {
      logger.info(`Getting schedule for date: ${date.toISOString().split('T')[0]}`);
      
      const startOfDay = new Date(date);
      startOfDay.setHours(0, 0, 0, 0);
      
      const endOfDay = new Date(date);
      endOfDay.setHours(23, 59, 59, 999);
      
      const query: any = {
        datetime: { $gte: startOfDay, $lte: endOfDay },
        status: { $ne: ReservationStatus.CANCELLED } // キャンセルされた予約は除外
      };
      
      // 特定の担当者のみ
      if (staffId) {
        query.staffId = staffId;
      }
      
      const reservations = await Reservation.find(query)
        .sort({ datetime: 1 })
        .exec();
      
      logger.info(`Found ${reservations.length} reservations for the specified date`);
      return reservations;
    } catch (error) {
      logger.error(`Failed to get daily schedule: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * 特定の時間枠が利用可能かチェックする
   * @param staffId 担当者ID
   * @param datetime 予約希望日時
   * @param duration 施術時間（分）
   * @param excludeReservationId 除外する予約ID（更新時に自分自身を除外）
   * @returns 利用可能かどうか
   */
  async checkAvailability(
    staffId: string,
    datetime: Date,
    duration: number,
    excludeReservationId?: string
  ): Promise<boolean> {
    try {
      logger.info(`Checking availability for staff ${staffId} at ${datetime}`);
      
      // 営業時間内かチェック
      const isWithinBusinessHours = await this.isWithinBusinessHours(datetime, duration);
      if (!isWithinBusinessHours) {
        logger.info('Requested time is outside business hours');
        return false;
      }
      
      // 予約開始時間と終了時間
      const startTime = new Date(datetime);
      const endTime = new Date(datetime);
      endTime.setMinutes(endTime.getMinutes() + duration);
      
      // 同じスタッフの既存予約を検索
      const query: any = {
        staffId,
        status: { $ne: ReservationStatus.CANCELLED }, // キャンセルされた予約は除外
        $or: [
          // 新しい予約の時間範囲が既存予約と重なるケース
          {
            datetime: { $lt: endTime },
            $expr: {
              $gt: {
                $add: ['$datetime', { $multiply: ['$duration', 60000] }]
              },
              startTime
            }
          }
        ]
      };
      
      // 更新の場合、自分自身の予約は除外
      if (excludeReservationId) {
        query._id = { $ne: excludeReservationId };
      }
      
      const conflictingReservations = await Reservation.find(query).exec();
      
      const isAvailable = conflictingReservations.length === 0;
      logger.info(`Time slot is ${isAvailable ? 'available' : 'not available'}`);
      
      return isAvailable;
    } catch (error) {
      logger.error(`Failed to check availability: ${error.message}`, { error });
      throw error;
    }
  }
  
  /**
   * 指定された時間が営業時間内かどうかをチェックする
   * @param datetime 予約希望日時
   * @param duration 施術時間（分）
   * @returns 営業時間内かどうか
   */
  private async isWithinBusinessHours(datetime: Date, duration: number): Promise<boolean> {
    // 営業時間の設定（例: 平日10:00-19:00、土日10:00-18:00）
    // 実際の実装では設定ファイルやデータベースから取得
    const dayOfWeek = datetime.getDay(); // 0: 日曜日, 1-5: 平日, 6: 土曜日
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
    
    const openHour = 10; // 10:00開始
    const closeHour = isWeekend ? 18 : 19; // 平日は19:00、週末は18:00まで
    
    const hour = datetime.getHours();
    const minute = datetime.getMinutes();
    
    // 開始時間が営業時間内か
    if (hour < openHour || hour >= closeHour) {
      return false;
    }
    
    // 終了時間が営業時間内か
    const endDatetime = new Date(datetime);
    endDatetime.setMinutes(endDatetime.getMinutes() + duration);
    
    if (endDatetime.getHours() > closeHour || 
        (endDatetime.getHours() === closeHour && endDatetime.getMinutes() > 0)) {
      return false;
    }
    
    return true;
  }
  
  /**
   * 予約を完了済みにする
   * @param reservationId 予約ID
   * @returns 更新された予約
   */
  async completeReservation(reservationId: string): Promise<Reservation> {
    try {
      logger.info(`Marking reservation ${reservationId} as completed`);
      
      // 予約の存在確認
      const reservation = await Reservation.findById(reservationId);
      if (!reservation) {
        logger.warn(`Reservation ${reservationId} not found`);
        throw new Error('予約が見つかりません');
      }
      
      // キャンセル済みの予約は完了に変更できない
      if (reservation.status === ReservationStatus.CANCELLED) {
        logger.warn(`Cannot complete cancelled reservation ${reservationId}`);
        throw new Error('キャンセルされた予約は完了に変更できません');
      }
      
      // すでに完了済みの場合
      if (reservation.status === ReservationStatus.COMPLETED) {
        logger.warn(`Reservation ${reservationId} is already completed`);
        return reservation;
      }
      
      // 予約を完了済みに変更
      reservation.status = ReservationStatus.COMPLETED;
      reservation.updatedAt = new Date();
      
      await reservation.save();
      
      logger.info(`Reservation ${reservationId} marked as completed`);
      return reservation;
    } catch (error) {
      logger.error(`Failed to complete reservation: ${error.message}`, { error });
      throw error;
    }
  }
}

export default new ReservationService();