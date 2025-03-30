import { Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn, CreateDateColumn, UpdateDateColumn } from 'typeorm';
import { IsDate, IsEnum, IsInt, IsNotEmpty, IsOptional, IsString, MaxLength, MinLength } from 'class-validator';
import { User } from './user_model';
import { Treatment } from './treatment_model';
import { Staff } from './staff_model';

/**
 * 予約ステータスの列挙型
 */
export enum ReservationStatus {
  RESERVED = '予約済み',
  CANCELLED = 'キャンセル済み',
  COMPLETED = '完了済み'
}

/**
 * 予約管理のデータモデル
 */
@Entity('reservations')
export class Reservation {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ name: 'user_id' })
  @IsNotEmpty({ message: 'ユーザーIDは必須です' })
  userId: string;

  @Column({ name: 'treatment_id' })
  @IsNotEmpty({ message: '施術IDは必須です' })
  treatmentId: string;

  @Column({ name: 'reservation_datetime', type: 'timestamp' })
  @IsDate({ message: '予約日時は有効な日付である必要があります' })
  @IsNotEmpty({ message: '予約日時は必須です' })
  reservationDatetime: Date;

  @Column({ name: 'treatment_duration', type: 'int' })
  @IsInt({ message: '施術時間は整数である必要があります' })
  @IsNotEmpty({ message: '施術時間は必須です' })
  treatmentDuration: number;

  @Column({ name: 'staff_id', nullable: true })
  @IsOptional()
  staffId: string | null;

  @Column({ 
    type: 'enum',
    enum: ReservationStatus,
    default: ReservationStatus.RESERVED
  })
  @IsEnum(ReservationStatus, { message: 'ステータスは予約済み、キャンセル済み、完了済みのいずれかである必要があります' })
  status: ReservationStatus;

  @Column({ name: 'notes', type: 'text', nullable: true })
  @IsOptional()
  @IsString({ message: '予約メモは文字列である必要があります' })
  @MaxLength(500, { message: '予約メモは500文字以内である必要があります' })
  notes: string | null;

  @CreateDateColumn({ name: 'created_at' })
  createdAt: Date;

  @UpdateDateColumn({ name: 'updated_at' })
  updatedAt: Date;

  // リレーションシップ
  @ManyToOne(() => User, user => user.reservations)
  @JoinColumn({ name: 'user_id' })
  user: User;

  @ManyToOne(() => Treatment, treatment => treatment.reservations)
  @JoinColumn({ name: 'treatment_id' })
  treatment: Treatment;

  @ManyToOne(() => Staff, staff => staff.reservations, { nullable: true })
  @JoinColumn({ name: 'staff_id' })
  staff: Staff | null;

  /**
   * 予約が有効かどうかを確認するメソッド
   * @returns 予約が有効な場合はtrue、そうでない場合はfalse
   */
  isValid(): boolean {
    const now = new Date();
    return this.status === ReservationStatus.RESERVED && 
           this.reservationDatetime > now;
  }

  /**
   * 予約をキャンセルするメソッド
   */
  cancel(): void {
    if (this.status !== ReservationStatus.COMPLETED) {
      this.status = ReservationStatus.CANCELLED;
    } else {
      throw new Error('完了済みの予約はキャンセルできません');
    }
  }

  /**
   * 予約を完了するメソッド
   */
  complete(): void {
    if (this.status === ReservationStatus.RESERVED) {
      this.status = ReservationStatus.COMPLETED;
    } else {
      throw new Error('予約済み状態の予約のみ完了にできます');
    }
  }
}

/**
 * 予約作成用のDTOインターフェース
 */
export interface CreateReservationDto {
  userId: string;
  treatmentId: string;
  reservationDatetime: Date;
  treatmentDuration: number;
  staffId?: string;
  notes?: string;
}

/**
 * 予約更新用のDTOインターフェース
 */
export interface UpdateReservationDto {
  reservationDatetime?: Date;
  treatmentDuration?: number;
  staffId?: string;
  status?: ReservationStatus;
  notes?: string;
}