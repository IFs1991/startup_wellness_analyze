import { mockCompanyDetails } from '@/lib/mocks/companies';
import type { Company, CompanyDetail } from '@/types/company';

interface GetCompaniesParams {
  page?: number;
  limit?: number;
  industry?: string;
  stage?: string;
  search?: string;
}

interface CompaniesResponse {
  companies: Company[];
  total: number;
  page: number;
  totalPages: number;
}

export const companiesApi = {
  getCompanies: async (params: GetCompaniesParams = {}): Promise<CompaniesResponse> => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const companies = Object.values(mockCompanyDetails).map(company => ({
      id: company.id,
      name: company.name,
      industry: company.industry,
      stage: company.stage,
      score: company.score,
      employees: company.employees,
      location: company.location,
      foundedYear: company.foundedYear,
    }));

    return {
      companies,
      total: companies.length,
      page: 1,
      totalPages: 1,
    };
  },

  getCompanyById: async (id: string): Promise<CompanyDetail> => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const company = mockCompanyDetails[id];
    if (!company) {
      throw new Error('Company not found');
    }
    
    return company;
  }
};